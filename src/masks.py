from typing import List


class MaskManager:
    def __init__(self, vocab: dict):
        """Initialize the mask manager with the model's vocabulary."""
        self.vocab = vocab

    def apply_mask(self, logits: List[float], allowed_words: List[str]):
        """Mask logits to only allow tokens that start with the exact allowed words."""
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

    def apply_number_mask(self, logits: List[float], len_param: int):
        """Specialized mask to only allow numeric characters and JSON structure symbols."""
        alwd = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        alwd = alwd | {".", "0", ",", "}", "-", " -"}

        if len_param == 0:
            alwd.remove(",")
        else:
            alwd.remove("}")

        valid_tokens_id = set()

        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", "")
            if all(char in alwd for char in clean_token):
                valid_tokens_id.add(token_id)

        for idx in range(len(logits)):
            if idx not in valid_tokens_id:
                logits[idx] = float("-inf")

    def apply_string_mask(self, logits: List[float]):
        """Mask to allow free text but strictly prevent newlines that break JSON format."""
        # In BPE encoding (Qwen), \n is 'Ċ', \r is 'č', and \t is 'ĉ'
        forbiden_tokens = {"\n", "\r", "\t", "Ċ", "č", "ĉ"}
        valid_token_id = set()

        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", "")
            # Check if NO forbidden character exists in the token string
            if not any(char in forbiden_tokens for char in clean_token):
                valid_token_id.add(token_id)

        for idx in range(len(logits)):
            if idx not in valid_token_id:
                logits[idx] = float("-inf")
