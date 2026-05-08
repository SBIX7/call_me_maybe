import json
import re
from typing import List

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDef


class JSONConstrainedDecoder:
    def __init__(self, llm: Small_LLM_Model, functions: List[FunctionDef]):
        """Initialize the decoder with the LLM and the allowed functions."""
        self.llm = llm
        self.functions = functions

        # Load vocabulary from the model
        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)

    def _apply_mask(self, logits: List[float], allowed_words: List[str]):
        """Mask logits to only allow tokens that start with the allowed words."""
        valid_tokens_id = set()
        for token_str, token_id in self.vocab.items():
            # Clean special characters from the token
            clean_token = token_str.replace("Ġ", " ")
            if token_str != "" and any(
                word.startswith(clean_token) for word in allowed_words
            ):
                valid_tokens_id.add(token_id)

        # Set non-valid tokens to negative infinity
        for idx in range(len(logits)):
            if idx not in valid_tokens_id:
                logits[idx] = float("-inf")

    def _apply_number_mask(self, logits: List[float], len_param: int):
        """Specialized mask to only allow numeric characters and specific symbols."""
        # Define the whitelist of characters for numbers
        alwd = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        alwd = alwd | {".", "0", ",", "}", "-"}
        if len_param == 0:
            alwd.remove(",")
        else:
            alwd.remove("}")
        valid_tokens_id = set()

        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", "")
            # Verify if the cleaned token is in the allowed characters list
            if clean_token in alwd:
                valid_tokens_id.add(token_id)

        # Apply the mask
        for idx in range(len(logits)):
            if idx not in valid_tokens_id:
                logits[idx] = float("-inf")

    def calculate_remainder(self, current_text: str, cible: str) -> tuple:
        """Calculate what part of the target string still needs to be generated."""
        rest_to_write = cible
        max_size = len(cible)
        for i in range(max_size, 0, -1):
            if current_text.endswith(rest_to_write[:i]):
                return rest_to_write[i:], i
        return cible, 0

    def _get_remainder(self, current_text: str, cibles: List[str]) -> list:
        """Get the remainders for a list of target strings."""
        potential_remainders = []
        for cible in cibles:
            remainder, score = self.calculate_remainder(current_text, cible)
            potential_remainders.append((remainder, score))
        return potential_remainders

    def decode(self, prompt: str) -> str:
        """Main decoding loop enforcing the JSON structure step by step."""
        i = 0
        generated_text = "{"
        state_machine = "STATE_WRITE_NAME_KEY"
        cible = ['"name": "']
        chosen_function = None
        parametres_restants = []
        current_key = None

        while True:
            # 1. Prepare input and get logits
            full_text = prompt + generated_text
            ids_2d = self.llm.encode(full_text)
            ids_1d = ids_2d[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids_1d)

            # ==========================================
            # 2. THE BRAIN: State Machine Transitions
            # ==========================================

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
                # Extract the chosen function name using regex
                match = re.search(r'"name": "(.*?)"', generated_text)
                if match:
                    func_name = match.group(1)
                    for f in self.functions:
                        if func_name == f.name:
                            chosen_function = f
                            break
                    # Prepare the remaining parameters to generate
                    if chosen_function:
                        parametres_restants = [
                            f'"{key}": ' for key in chosen_function.parameters.keys()
                        ]
                        if parametres_restants:
                            cible = [parametres_restants[0]]

            if state_machine == "STATE_WRITE_ARG_KEY" and generated_text.endswith(
                tuple(cible)
            ):
                state_machine = "STATE_WRITE_ARG_VALUE"
                # Extraction de la clé pour que les Muscles puissent la lire
                current_key = cible[0].strip(": ").strip('"')
                parametres_restants.pop(0)

            if state_machine == "STATE_WRITE_ARG_VALUE" and (
                generated_text.endswith(",") or generated_text.endswith("}")
            ):
                if generated_text.endswith(",") and parametres_restants != []:
                    state_machine = "STATE_WRITE_ARG_KEY"
                    cible = [parametres_restants[0]]
                elif generated_text.endswith("}"):
                    state_machine = "STATE_END_JSON"
                    cible = ["}}"]

            if state_machine == "STATE_END_JSON" and (generated_text.endswith("}}")):
                break
            # ==========================================
            # 3. THE MUSCLES: Applying Masks & Execution
            # ==========================================

            if state_machine == "STATE_WRITE_ARG_VALUE":
                # Si c'est un nombre, on applique le masque numérique
                if (
                    chosen_function
                    and chosen_function.parameters[current_key].type == "number"
                ):
                    self._apply_number_mask(logits, len(parametres_restants))

            else:
                # Pour tous les autres états, on force la 'cible' exacte
                targets = cible
                permitted_words = self._get_remainder(generated_text, targets)
                if permitted_words:
                    higher_score = max([score for _, score in permitted_words])
                    permitted_words = [
                        word for word, score in permitted_words if score == higher_score
                    ]
                self._apply_mask(logits, permitted_words)

            # Generate token for standard states
            best_token_id = logits.index(max(logits))
            best_token_str = self.llm.decode([best_token_id])
            generated_text += best_token_str

            # Log current status
            print(f"Tour {i} | Etat: {state_machine} | Texte : {generated_text}")

            # Safety break
            if i == 200:
                break
            i += 1
        if i == 0:
            print(prompt)
        return generated_text
