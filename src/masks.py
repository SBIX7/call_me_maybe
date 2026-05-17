"""
Provides token masking utilities to enforce JSON grammar during LLM generation.
"""

from typing import List, Dict


class MaskManager:
    """
    Manages vocabulary masking to restrict LLM token
    generation based on required types.
    """
    def __init__(self, vocab: Dict[str, int]) -> None:
        """Initialize the mask manager with the model's vocabulary."""
        self.vocab = vocab

    def apply_mask(self, logits: List[float], allowed_wrds: List[str]) -> None:
        """Restrict logits to tokens that strictly match
        allowed target prefixes."""
        valid_tokens_id = set()
        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", " ")
            if token_str != "" and any(
                word.startswith(clean_token) for word in allowed_wrds
            ):
                valid_tokens_id.add(token_id)

        for idx in range(len(logits)):
            if idx not in valid_tokens_id:
                logits[idx] = float("-inf")

    def apply_number_mask(self, logits: List[float], len_param: int) -> None:
        """Restrict logits to numeric characters and
        valid JSON structural symbols."""
        alwd = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        alwd = alwd | {".", "0", ",", "}", "-", " ", "+", "−"}

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

    def apply_string_mask(self, logits: List[float]) -> None:
        """Allow free text generation while explicitly
        blocking newline characters."""
        forbiden_tokens = {"\n", "\r", "\t", "Ċ", "č", "ĉ"}
        valid_token_id = set()

        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", "")
            if not any(char in forbiden_tokens for char in clean_token):
                valid_token_id.add(token_id)

        for idx in range(len(logits)):
            if idx not in valid_token_id:
                logits[idx] = float("-inf")
