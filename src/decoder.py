from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDef
from typing import List
import json


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
        for id in range(len(logits)):
            if id not in valid_tokens_id:
                logits[id] = float("-inf")

    def calculate_remainder(self, current_text: str, cible: str) -> str:
        rest_to_write = cible
        max_size = len(cible)
        for i in range(max_size, 0, -1):
            if current_text.endswith(rest_to_write[:i]):
                return rest_to_write[i:], i
        return cible, 0

    def _get_remainder(self, current_text: str, cibles: List[str]) -> List[str, int]:
        potential_remainders = []
        for cible in cibles:
            remainder, score = self.calculate_remainder(current_text, cible)
            potential_remainders.append((remainder, score))
        return potential_remainders

    def decode(self, prompt: str) -> str:
        i = 0
        generated_text = "{"
        while True:
            full_text = prompt + generated_text
            ids_2d = self.llm.encode(full_text)
            ids_1d = ids_2d[0].tolist()
            logits = self.llm.get_logits_from_input_ids(ids_1d)
            if '"name": "' not in generated_text:
                targets = ['"name": "']
                permitted_words = self._get_remainder(generated_text, targets)
                higher_score = max([score for _, score in permitted_words])
                permitted_words = [
                    word for word, score in permitted_words if score == higher_score
                ]
                self._apply_mask(logits, permitted_words)
            elif '", "' not in generated_text:
                targets = [elem.name + '", "' for elem in self.functions]
                permitted_words = self._get_remainder(generated_text, targets)
                higher_score = max([score for _, score in permitted_words])
                permitted_words = [
                    word for word, score in permitted_words if score == higher_score
                ]
                self._apply_mask(logits, permitted_words)
            elif '"arguments": {' not in generated_text:
                targets = ['arguments": {']
                permitted_words = self._get_remainder(generated_text, targets)
                higher_score = max([score for _, score in permitted_words])
                permitted_words = [
                    word for word, score in permitted_words if score == higher_score
                ]
                self._apply_mask(logits, permitted_words)
            best_token_id = logits.index(max(logits))
            best_token_str = self.llm.decode([best_token_id])
            generated_text += best_token_str
            print(f"Tour {i} : {generated_text}")
            print(f"Reste autorisé : {permitted_words}")
            print(f"Token choisi par l'IA : '{best_token_str}' (ID: {best_token_id})")
            if i == 200:
                break
            i += 1
        return generated_text
