*This project has been created as part of the 42 curriculum by msbii.*

# Call Me Maybe: Function Calling in LLMs

## Description
This project bridges the gap between natural language understanding and machine-executable structured output. Using a small, local Large Language Model (Qwen3-0.6B), this tool accurately translates user prompts into strictly validated JSON function calls. It demonstrates that with precise algorithmic control (Constrained Decoding), a 0.5B/0.6B parameter model can achieve the structural and semantic reliability of much larger production models.

## Algorithm Explanation (Constrained Decoding)
Relying on a small LLM to naturally output valid JSON via prompting is highly unreliable, frequently leading to formatting errors or schema hallucinations. To guarantee 100% valid JSON and absolute schema compliance, this project implements a **Constrained Decoding** engine using a custom State Machine and a Prefix-Matching Mask Manager.

The generation process operates dynamically token-by-token:
1. **The Brain (State Machine):** Analyzes the currently generated text history to track the current structural state of the JSON (e.g., `NAME_KEY`, `FUNC_NAME`, `PARAMS_KEY`, `ARG_KEY`, `STR_CONTENT`, `NUM_VAL`, `BOOL_VAL`, `ENUM_VAL`).
2. **The Suffix Evaluator (Prefix-Matching):** Determines the exact remainder of the target string that needs to be completed by calculating the maximum overlap between the current text and the expected JSON tokens.
3. **The Arms (Logits Masking):** Before the model performs its next token selection, the `MaskManager` intercepts the raw output token probabilities (`logits`). Any token in the vocabulary that violates the JSON syntax rules, breaches the expected types, or fails to match the predefined enum values has its logit set to negative infinity (`float("-inf")`).
4. **The Selection:** The model is mathematically forced to select its next token exclusively from the strictly allowed subset of valid continuations.

This continuous runtime intervention guarantees that the model cannot physically produce non-parseable output, syntax traps, or malformed data types.

## Design Decisions
* **Separation of Concerns:** The decoding logic is strictly divided into two specialized modules: state transitions (`_update_state`) and logit modification (`_apply_masks`). This modularity isolates structural logic from vocabulary manipulation, making complex typing and schema enforcement straightforward to maintain.
* **Strict Schema & Enum Enforcement:** When a function parameter specifies an `enum`, the state machine transitions to an `ENUM_VAL` state. The mask manager then restricts token choices entirely to the allowed string values. Similarly, required keys are tracked using a remaining parameters list (`p_left`), ensuring no required fields are omitted.
* **Typing and Linting Rigor:** To maintain a high standard of code quality and sustainability, the project adheres strictly to PEP 8 standard specifications via `flake8` and enforces full static type checking with `mypy --strict`. All methods use explicit type hints and context managers for safe file handling.

## Performance Analysis
* **Execution Speed:** The solution processes a standard evaluation suite well within the strict 5-minute project limit. Thanks to algorithmic optimizations, a complete execution run maintains stable performance across varying prompt lengths.
* **Encoding & Token-IDs Optimization:** Instead of iteratively re-tokenizing the entire accumulated text string at each generation step (which causes an exponential $O(N^2)$ slowdown), the `system_prompt` and user prompt are encoded exactly once into a 1D list of token IDs before the generation loop. Subsequent tokens are simply appended directly to this list (`ids_1d.append(best_tok)`). This leverages the underlying transformer's KV Cache efficiently, reducing processing overhead.
* **Reliability & Validity:** The solution achieves a **100% JSON parsing success rate** across all valid test cases. Prose, conversational text, and conversational padding are entirely eliminated from the output.

## Challenges Faced
1. **Performance Bottlenecks:** Iterative string tokenization originally introduced dramatic processing delays. Transitioning to a native 1D list operation and relying on sequential token ID appending resolved the overhead, bringing the execution time down.
2. **Handling Subwords and Space Identifiers (Ġ):** The model's vocabulary represents spaces using specific character encodings (like `Ġ`). When executing prefix-matching for strict target tokens, a careful string cleaning strategy (`.replace("Ġ", " ")` vs `.replace("Ġ", "")`) had to be devised to differentiate between rigid JSON structural whitespace and flexible numeric/text padding without breaking native `json.loads` compatibility.
3. **Strict MyPy Alignment:** Conforming to `mypy --strict` required precise generic declarations across the state machine's multi-variable state transitions, demanding disciplined handling of optional types and dynamic structure inputs.

## Testing Strategy
The implementation was rigorously validated using a comprehensive suite of edge cases and adversarial inputs:
* **Syntax and Escape Traps:** Prompts containing inner escaped quotes (e.g., `\"hello\"`), special characters, and empty strings were tested to verify that the negative lookbehind regex (`(?<!\\\\)"`) safely identifies the authentic closing boundaries of a JSON string.
* **Type Forcing and Sign Preservation:** Negative values, float representations (forcing `.0` suffixes via system rules), and large numbers were evaluated to ensure the numeric mask safely allows `-`, `.`, and digits while maintaining mathematical validity.
* **Enum Traps:** Prompts providing non-compliant options were passed to confirm that the constrained decoder restricts choices exclusively to valid tokens, preventing data corruption.
* **Robustness & Degradation:** Missing or malformed input files (such as invalid JSON configurations) are intercepted gracefully using structured `try-except` blocks, printing descriptive error messages to `sys.stderr` and terminating safely without unhandled exceptions or crashes.

## Instructions

### Installation
Synchronize the virtual environment and install the required dependencies (`pydantic`, `numpy`) using `uv`:
```bash
make install
```

### Execution
To run the function calling pipeline with default paths (`data/input/` and `data/output/`):
```bash
make run
```

Alternatively, you can run the program with custom file paths by providing explicit arguments:
```bash
uv run python -m src --functions_definition custom_definitions.json --input custom_tests.json --output custom_results.json
```

## Example usage

### Input Prompt
```text
"What is the sum of 40 and 2?"
```

### Generated JSON Output (Saved to `data/output/function_calls.json`)
```json
[
    {
        "prompt": "What is the sum of 40 and 2?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 40.0,
            "b": 2.0
        }
    }
]
```

## Resources
* **References:** * [Pydantic Core Documentation](https://docs.pydantic.dev/)
  * [Python Regular Expressions (re module) Guides](https://docs.python.org/3/library/re.html)
  * [Understanding LLM Logits and Constrained Generation](https://github.com/QwenLM/Qwen3)
* **AI Usage:** Artificial Intelligence platforms (Gemini and ChatGPT) were utilized during development as advanced interactive technical references. They assisted in clarifying low-level Transformer behaviors (such as KV Caching mechanisms and self-attention tensor mechanics), designing regular expressions (specifically the negative lookbehind pattern used to detect string boundaries), and reviewing the code layout to guarantee strict adherence to PEP 257 docstring compliance. All core algorithmic structures, state machines, and masking frameworks were uniquely architected, manually written, and verified to ensure full conceptual understanding.