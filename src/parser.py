import json
from src.models import FunctionDef, Prompt
from typing import List


def load_functions_definitions(filepath: str) -> List[FunctionDef]:
    validated_functions: List[FunctionDef] = []
    with open(filepath, "r") as f:
        raw_data = json.load(f)
    for elem in raw_data:
        function = FunctionDef(**elem)
        validated_functions.append(function)
    return validated_functions


def load_prompt(filepath: str) -> List[Prompt]:
    valid_prompts: List[Prompt] = []
    with open(filepath, "r") as f:
        raw_data = json.load(f)
    for elem in raw_data:
        prompt = Prompt(**elem)
        valid_prompts.append(prompt)
    return valid_prompts
