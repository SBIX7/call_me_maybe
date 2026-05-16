from typing import List, Tuple


def calculate_remainder(curr_txt: str, cible: str) -> Tuple[str, int]:
    """Calculate what part of the target string still needs to be generated."""
    rest_to_write = cible
    max_size = len(cible)
    for i in range(max_size, 0, -1):
        if curr_txt.endswith(rest_to_write[:i]):
            return rest_to_write[i:], i
    return cible, 0


def get_remainder(curr_txt: str, cibles: List[str]) -> List[Tuple[str, int]]:
    """Get the remainders for a list of target strings."""
    potential_remainders = []
    for cible in cibles:
        remainder, score = calculate_remainder(curr_txt, cible)
        potential_remainders.append((remainder, score))
    return potential_remainders
