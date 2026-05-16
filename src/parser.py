import json
import sys
from src.models import FunctionDef, Prompt
from typing import List


def load_functions_definitions(filepath: str) -> List[FunctionDef]:
    validated_functions: List[FunctionDef] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for elem in raw_data:
            function = FunctionDef(**elem)
            validated_functions.append(function)
        return validated_functions
    except Exception as e:
        print(
            f"Error reading functions file ({filepath}): {e}",
            file=sys.stderr
        )
        sys.exit(1)


def load_prompt(filepath: str) -> List[Prompt]:
    valid_prompts: List[Prompt] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for elem in raw_data:
            prompt = Prompt(**elem)
            valid_prompts.append(prompt)
        return valid_prompts
    except Exception as e:
        print(
            f"Error reading prompts file ({filepath}): {e}",
            file=sys.stderr
        )
        sys.exit(1)
