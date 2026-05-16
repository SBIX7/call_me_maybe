import argparse
import json
import os
import sys
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
    
    try:
        functions = load_functions_definitions(args.functions_definition)
        prompts = load_prompt(args.input)
    except Exception as e:
        print(f"Fatal error loading files: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        llm = Small_LLM_Model()
        decoder = JSONConstrainedDecoder(llm, functions)
    except Exception as e:
        print(f"Error initializing LLM or Decoder: {e}", file=sys.stderr)
        sys.exit(1)

    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    results = []
    
    # --- CORRECTION: Gestion propre de Ctrl+C et Ctrl+D ---
    try:
        for i in range(len(prompts)):
            try:
                raw_output = decoder.decode(prompts[i].model_dump()['prompt'])
                sanitized_output = raw_output.replace("\\", "\\\\").replace('\\\\"', '\\"')
                parsed_json = json.loads(sanitized_output)
                results.append(parsed_json)
            except json.JSONDecodeError as e:
                print(f"\nWarning: LLM generated invalid JSON for prompt: {prompts[i].prompt}\nError: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"\nUnexpected error decoding prompt {i}: {e}", file=sys.stderr)
                continue
    except (KeyboardInterrupt, EOFError):
        print("\n\nExecution interrupted by user (Ctrl+C/Ctrl+D). Saving progress...", file=sys.stderr)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Fatal error saving file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
