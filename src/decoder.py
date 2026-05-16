import json
import re
from typing import List, Any, Tuple

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDef
from src.utils import get_remainder
from src.masks import MaskManager


class JSONConstrainedDecoder:
    def __init__(self, llm: Small_LLM_Model, funcs: List[FunctionDef]) -> None:
        self.llm = llm
        self.functions = funcs

        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.masker = MaskManager(vocab)

    def decode(self, prompt: str) -> str:
        """ENGINE: Coordinates the main decoding loop."""
        sys_prompt = self._build_system_prompt()
        sys_prompt += f"\nPrompt: {prompt}<|im_end|>\n<|im_start|>assistant\n"

        gen_text = '{"prompt": ' + json.dumps(prompt) + ", "
        print(gen_text, end="", flush=True)

        state = "NAME_KEY"
        cible = ['"name": "']
        c_func = None
        p_left: List[str] = []
        c_key = ""
        s_start = 0

        # Optimization: Encode the whole prompt only once
        full_text = sys_prompt + gen_text
        ids_2d = self.llm.encode(full_text)
        ids_1d = ids_2d[0].tolist()

        for _ in range(400):
            logits = self.llm.get_logits_from_input_ids(ids_1d)

            # Brain makes decisions
            state, cible, c_func, p_left, c_key, s_start, gen_text = \
                self._update_state(
                    gen_text, state, cible, c_func,
                    p_left, c_key, s_start
                )

            if state == "END_JSON" and gen_text.endswith("}}"):
                break

            # Arms restrict AI tokens
            self._apply_masks(
                logits, state, cible, gen_text, c_key, c_func, len(p_left)
            )

            best_tok = logits.index(max(logits))
            best_str = self.llm.decode([best_tok])

            gen_text += best_str
            ids_1d.append(best_tok)  # Speed optimization
            print(best_str, end="", flush=True)

        print()
        return gen_text

    def _build_system_prompt(self) -> str:
        prompt = (
            "<|im_start|>system\n"
            "You are a precise data extraction AI. "
            "CRITICAL: You MUST output all numbers as floats. "
            "(e.g., 2.0 instead of 2). "
            "You MUST preserve negative signs. "
            "If prompt says '-2', output -2.0.<|im_end|>\n"
            "<|im_start|>user\n"
            "Available functions:\n"
        )
        for f in self.functions:
            prompt += f"- {f.name}: {f.description}\n"
        prompt += (
            "\n--- EXAMPLE ---\n"
            "Prompt: What is the addition of -5 and 4?\n"
            'Output: {"prompt": "What is the addition of -5 and 4?", '
            '"name": "fn_add_numbers", "parameters": {"a": -5.0, "b": 4.0}}\n'
            "---------------\n"
        )
        return prompt

    def _update_state(
        self, gen_txt: str, state: str, cible: List[str],
        c_func: Any, p_left: List[str], c_key: str, s_start: int
    ) -> Tuple[str, List[str], Any, List[str], str, int, str]:
        """BRAIN: State Machine transitions logic."""

        if state == "NAME_KEY" and gen_txt.endswith(tuple(cible)):
            state = "FUNC_NAME"
            cible = [f.name + '", "' for f in self.functions]

        if state == "FUNC_NAME" and gen_txt.endswith(tuple(cible)):
            state = "PARAMS_KEY"
            cible = ['parameters": {']

        if state == "PARAMS_KEY" and gen_txt.endswith(tuple(cible)):
            state = "ARG_KEY"
            match = re.search(r'"name": "(.*?)"', gen_txt)
            if match:
                f_name = match.group(1)
                for f in self.functions:
                    if f_name == f.name:
                        c_func = f
                        break
                if c_func:
                    p_left = [f'"{k}":' for k in c_func.parameters.keys()]
                    if p_left:
                        cible = [p_left[0]]
                    else:
                        state = "END_JSON"
                        cible = ["}}"]

        if state == "ARG_KEY" and gen_txt.endswith(tuple(cible)):
            c_key = cible[0].replace(", ", "").replace(",", "").strip(': "')
            p_left.pop(0)

            if c_func:
                param_def = c_func.parameters[c_key]

                if param_def.enum:
                    state = "ENUM_VAL"
                    cible = [f'"{val}"' for val in param_def.enum]
                elif param_def.type == "string":
                    state = "STR_START"
                    cible = ['"']
                elif param_def.type == "boolean":
                    state = "BOOL_VAL"
                    cible = ["true", "false", " true", " false"]
                else:
                    state = "NUM_VAL"

        if state == "STR_START" and gen_txt.endswith('"'):
            state = "STR_CONTENT"
            s_start = len(gen_txt)

        if state == "STR_CONTENT":
            new_content = gen_txt[s_start:]
            match = re.search(r'(?<!\\)"', new_content)
            if match:
                quote_pos = s_start + match.start()
                gen_txt = gen_txt[: quote_pos + 1]
                if p_left:
                    state = "ARG_KEY"
                    cible = [", " + p_left[0]]
                else:
                    state = "END_JSON"
                    cible = ["}}"]

        if state in ("BOOL_VAL", "ENUM_VAL") \
           and gen_txt.endswith(tuple(cible)):
            if p_left:
                state = "ARG_KEY"
                cible = [", " + p_left[0]]
            else:
                state = "END_JSON"
                cible = ["}}"]

        if state == "NUM_VAL" and (
            gen_txt.endswith(",") or gen_txt.endswith("}")
        ):
            if gen_txt.endswith(",") and p_left:
                state = "ARG_KEY"
                cible = [p_left[0]]
            elif gen_txt.endswith("}"):
                state = "END_JSON"
                cible = ["}}"]

        return state, cible, c_func, p_left, c_key, s_start, gen_txt

    def _apply_masks(
            self, logits: List[float], state: str, cible: List[str],
            gen_txt: str, c_key: str, c_func: Any, n_left: int) -> None:
        """ARMS: Restrict AI tokens based on state."""
        if state == "NUM_VAL":
            if c_func and c_func.parameters[c_key].type == "number":
                self.masker.apply_number_mask(logits, n_left)
        elif state == "STR_CONTENT":
            self.masker.apply_string_mask(logits)
        else:
            targets = cible
            remainders = get_remainder(gen_txt, targets)
            valid_words: List[str] = []

            if remainders:
                higher_score = max([s for _, s in remainders])
                valid_words = [w for w, s in remainders if s == higher_score]

            self.masker.apply_mask(logits, valid_words)
