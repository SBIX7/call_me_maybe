import json
import re
from typing import List

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDef
from src.utils import get_remainder
from src.masks import MaskManager


class JSONConstrainedDecoder:
    def __init__(self, llm: Small_LLM_Model, functions: List[FunctionDef]):
        """Initialize the decoder with the LLM and the allowed functions."""
        self.llm = llm
        self.functions = functions

        # Load vocabulary from the model and initialize the MaskManager
        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.masker = MaskManager(vocab)

    def decode(self, prompt: str) -> str:
        """Main decoding loop enforcing the JSON structure step by step."""

        # 1. Prepare the System Prompt (Context & Instructions for the LLM)
        system_prompt = "You are a helpful AI assistant. Extract the correct function call.\nAvailable functions:\n"
        system_prompt += "CRITICAL: Convert written negative numbers like 'negative 2' into mathematical symbols like '-2'.\n"
        for function in self.functions:
            system_prompt += f"- {function.name}: {function.description}\n"
        system_prompt += f"\nUser: {prompt}\nAssistant:\n"

        # 2. Initialize State Machine Variables
        i = 0

        # json.dumps(prompt) is CRITICAL here to escape internal quotes properly!
        generated_text = '{"prompt": ' + json.dumps(prompt) + ", "

        state_machine = "STATE_WRITE_NAME_KEY"
        cible = ['"name": "']
        chosen_function = None
        parametres_restants = []
        current_key = None
        string_start_len = 0

        while True:
            # 3. Prepare input and get logits
            full_text = system_prompt + generated_text

            ids_2d = self.llm.encode(full_text)
            ids_1d = ids_2d[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids_1d)

            # ---------------------------------------------------------
            # 4. STATE MACHINE TRANSITIONS (The Brain)
            # ---------------------------------------------------------

            if state_machine == "STATE_WRITE_NAME_KEY" and generated_text.endswith(
                tuple(cible)
            ):
                state_machine = "STATE_WRITE_FUNC_NAME"
                cible = [f.name + '", "' for f in self.functions]

            if state_machine == "STATE_WRITE_FUNC_NAME" and generated_text.endswith(
                tuple(cible)
            ):
                state_machine = "STATE_WRITE_PARAMS_KEY"
                cible = ['parameters": {']

            if state_machine == "STATE_WRITE_PARAMS_KEY" and generated_text.endswith(
                tuple(cible)
            ):
                state_machine = "STATE_WRITE_ARG_KEY"

                match = re.search(r'"name": "(.*?)"', generated_text)
                if match:
                    func_name = match.group(1)
                    for f in self.functions:
                        if func_name == f.name:
                            chosen_function = f
                            break

                    if chosen_function:
                        parametres_restants = [
                            f'"{key}": ' for key in chosen_function.parameters.keys()
                        ]
                        if parametres_restants:
                            cible = [parametres_restants[0]]

            if state_machine == "STATE_WRITE_ARG_KEY" and generated_text.endswith(
                tuple(cible)
            ):
                current_key = cible[0].replace(", ", "").replace(",", "").strip(': "')
                parametres_restants.pop(0)

                if (
                    chosen_function
                    and chosen_function.parameters[current_key].type == "string"
                ):
                    state_machine = "STATE_WRITE_STRING_START"
                    cible = ['"']
                else:
                    state_machine = "STATE_WRITE_ARG_VALUE"

            # --- STRING HANDLING STATES ---

            if state_machine == "STATE_WRITE_STRING_START" and generated_text.endswith(
                '"'
            ):
                state_machine = "STATE_WRITE_STRING_CONTENT"
                string_start_len = len(generated_text)

            if state_machine == "STATE_WRITE_STRING_CONTENT":
                new_content = generated_text[string_start_len:]

                # MAGIC REGEX: Finds a quote that is NOT preceded by a backslash
                # This ignores escaped quotes inside the string (like \")
                match = re.search(r'(?<!\\)"', new_content)

                if match:
                    # Truncate at the exact position of the real closing quote
                    quote_pos = string_start_len + match.start()
                    generated_text = generated_text[: quote_pos + 1]

                    # Take back control of the state machine
                    if parametres_restants:
                        state_machine = "STATE_WRITE_ARG_KEY"
                        cible = [", " + parametres_restants[0]]
                    else:
                        state_machine = "STATE_END_JSON"
                        cible = ["}}"]

            # --- NUMBER HANDLING STATE ---

            if state_machine == "STATE_WRITE_ARG_VALUE" and (
                generated_text.endswith(",") or generated_text.endswith("}")
            ):
                if generated_text.endswith(",") and parametres_restants:
                    state_machine = "STATE_WRITE_ARG_KEY"
                    cible = [parametres_restants[0]]
                elif generated_text.endswith("}"):
                    state_machine = "STATE_END_JSON"
                    cible = ["}}"]

            # --- VICTORY CONDITION ---
            if state_machine == "STATE_END_JSON" and generated_text.endswith("}}"):
                break

            # ---------------------------------------------------------
            # 5. EXECUTION: APPLYING MASKS
            # ---------------------------------------------------------

            if state_machine == "STATE_WRITE_ARG_VALUE":
                if (
                    chosen_function
                    and chosen_function.parameters[current_key].type == "number"
                ):
                    self.masker.apply_number_mask(logits, len(parametres_restants))

            elif state_machine == "STATE_WRITE_STRING_CONTENT":
                self.masker.apply_string_mask(logits)

            else:
                targets = cible
                permitted_words = get_remainder(generated_text, targets)

                if permitted_words:
                    higher_score = max([score for _, score in permitted_words])
                    permitted_words = [
                        word for word, score in permitted_words if score == higher_score
                    ]
                self.masker.apply_mask(logits, permitted_words)

            best_token_id = logits.index(max(logits))
            best_token_str = self.llm.decode([best_token_id])
            generated_text += best_token_str
            print(generated_text)
            # Increased to 400 because Regex generation can take many tokens
            if i == 400:
                break
            i += 1

        return generated_text
