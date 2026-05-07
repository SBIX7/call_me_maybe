import json
import re
from typing import List

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDef


class JSONConstrainedDecoder:
    def __init__(self, llm: Small_LLM_Model, functions: List[FunctionDef]):
        self.llm = llm
        self.functions = functions
        vocab_path = llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)

    def _apply_mask(self, logits: List[float], allowed_words: List[str]):
        valid_tokens_id = set()
        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", " ")
            if token_str != "" and any(
                word.startswith(clean_token) for word in allowed_words
            ):
                valid_tokens_id.add(token_id)
        for idx in range(len(logits)):
            if idx not in valid_tokens_id:
                logits[idx] = float("-inf")

    def calculate_remainder(self, current_text: str, cible: str) -> tuple:
        rest_to_write = cible
        max_size = len(cible)
        for i in range(max_size, 0, -1):
            if current_text.endswith(rest_to_write[:i]):
                return rest_to_write[i:], i
        return cible, 0

    def _get_remainder(self, current_text: str, cibles: List[str]) -> list:
        potential_remainders = []
        for cible in cibles:
            remainder, score = self.calculate_remainder(current_text, cible)
            potential_remainders.append((remainder, score))
        return potential_remainders

    def decode(self, prompt: str) -> str:
        i = 0
        generated_text = "{"
        state_machine = "STATE_WRITE_NAME_KEY"
        cible = ['"name": "']
        chosen_function = None
        parametres_restants = []

        while True:
            full_text = prompt + generated_text
            ids_2d = self.llm.encode(full_text)
            ids_1d = ids_2d[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids_1d)

            if (
                state_machine == "STATE_WRITE_NAME_KEY"
                and generated_text.endswith(tuple(cible))
            ):
                state_machine = "STATE_WRITE_FUNC_NAME"
                cible = [f.name + '", "' for f in self.functions]

            elif (
                state_machine == "STATE_WRITE_FUNC_NAME"
                and generated_text.endswith(tuple(cible))
            ):
                state_machine = "STATE_WRITE_PARAMS_KEY"
                cible = ['parameters": {']

            elif (
                state_machine == "STATE_WRITE_PARAMS_KEY"
                and generated_text.endswith(tuple(cible))
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

            elif (
                state_machine == "STATE_WRITE_ARG_KEY"
                and generated_text.endswith(tuple(cible))
            ):
                state_machine = "STATE_WRITE_ARG_VALUE"
                if chosen_function:
                    if chosen_function.parameters.get(cible[0]) == "number":
                        alwd = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
                        alwd += [".", "0", ",", "}", "-"]
                        self._apply_mask(logits, alwd)
                        best_token_id = logits.index(max(logits))
                        best_token_str = self.llm.decode([best_token_id])
                        generated_text += best_token_str

                        print(f"Tour {i} | Etat: {state_machine} | Texte : {generated_text}")

                        if i == 200:
                            break
                        i += 1


                        continue
                

            targets = cible
            permitted_words = self._get_remainder(generated_text, targets)

            if permitted_words:
                higher_score = max([score for _, score in permitted_words])
                permitted_words = [
                    word
                    for word, score in permitted_words
                    if score == higher_score
                ]

            self._apply_mask(logits, permitted_words)

            best_token_id = logits.index(max(logits))
            best_token_str = self.llm.decode([best_token_id])
            generated_text += best_token_str

            print(f"Tour {i} | Etat: {state_machine} | Texte : {generated_text}")

            if i == 200:
                break
            i += 1

        return generated_text
