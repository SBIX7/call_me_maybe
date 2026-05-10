from typing import List


def calculate_remainder(current_text: str, cible: str) -> tuple:
    """Calculate what part of the target string still needs to be generated."""
    rest_to_write = cible
    max_size = len(cible)
    for i in range(max_size, 0, -1):
        if current_text.endswith(rest_to_write[:i]):
            return rest_to_write[i:], i
    return cible, 0


def get_remainder(current_text: str, cibles: List[str]) -> list:
    """Get the remainders for a list of target strings."""
    potential_remainders = []
    for cible in cibles:
        remainder, score = calculate_remainder(current_text, cible)
        potential_remainders.append((remainder, score))
    return potential_remainders
