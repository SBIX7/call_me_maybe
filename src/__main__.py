import argparse
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
    functions = load_functions_definitions(args.functions_definition)
    prompts = load_prompt(args.input)
    llm = Small_LLM_Model()
    decoder = JSONConstrainedDecoder(llm, functions)
    print(decoder.decode(prompts[3].model_dump()['prompt']))


if __name__ == "__main__":
    main()
