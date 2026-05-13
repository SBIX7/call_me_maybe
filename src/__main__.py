import argparse
import json
import os
from src.parser import load_functions_definitions, load_prompt
from llm_sdk.llm_sdk import Small_LLM_Model
from src.decoder import JSONConstrainedDecoder


def main():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        type=str,
        help="Path to JSON file containing functions definitons",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        type=str,
        help="Path to JSON file containing prompts",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        type=str,
        help="Path to where final JSON will be saved",
    )
    args = parser.parse_args()
    final_json = ""
    functions = load_functions_definitions(args.functions_definition)
    prompts = load_prompt(args.input)
    llm = Small_LLM_Model()
    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    i = 0
    decoder = JSONConstrainedDecoder(llm, functions)
    for i in range(len(prompts)):
        final_json += decoder.decode(prompts[i].model_dump()['prompt'])
        final_json += ", "
        if i != len(prompts) - 1:
            final_json = final_json.rstrip("}")
    final_json = "[" + final_json.rstrip(", ") + "]"
    with open(args.output, "w") as f:
        f.write(final_json)


if __name__ == "__main__":
    main()
