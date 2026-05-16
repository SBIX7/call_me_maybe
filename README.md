*This project has been created as part of the 42 curriculum by msbii.*

# Call Me Maybe: Function Calling in LLMs

## Description
This project bridges the gap between natural language understanding and machine-executable structured output. Using a small, local Large Language Model (Qwen3-0.6B), this tool accurately translates user prompts into strictly validated JSON function calls. It demonstrates that with precise algorithmic control (Constrained Decoding), a 0.6B parameter model can achieve the structural reliability of much larger production models.

## Algorithm Explanation (Constrained Decoding)
Relying on a small LLM to naturally output valid JSON via prompting is highly unreliable. To guarantee 100% valid JSON and schema compliance, this project implements a **Constrained Decoding** engine using a State Machine and a Mask Manager.

The process operates token-by-token:
1. **The Brain (State Machine):** Analyzes the currently generated text to determine the required JSON structural state (e.g., `NAME_KEY`, `ENUM_VAL`, `NUM_VAL`).
2. **The Arms (Logits Masking):** Based on the state, the algorithm modifies the model's output probabilities (`logits`). Any token that violates the JSON syntax or the expected function schema (types, enums) has its probability set to negative infinity (`float("-inf")`).
3. **The Engine (Token Selection):** The model is then forced to pick the most probable token only from the strictly allowed subset. 

This guarantees that the model cannot physically hallucinate incorrect keys, invalid data types, or non-compliant enum values.

## Design Decisions
* **Separation of Concerns:** The decoding logic was refactored into distinct responsibilities: state transitions (`_update_state`) and logit masking (`_apply_masks`). This modularity makes debugging strict typing and enum matching much easier.
* **Strict Enum Enforcement:** To ensure arguments strictly match predefined options, the decoder dynamically transitions to an `ENUM_VAL` state when it detects a Pydantic `enum` parameter. This restricts token generation entirely to the specific words allowed in the function definition.
* **Typing and Linting Rigor:** The project adheres to strict PEP 8 norms (`flake8`) and uncompromising static type checking (`mypy --strict`). 

## Performance Analysis
* **Execution Speed:** Processing an extensive stress-test suite of 20 complex prompts takes under 8 minutes, meaning a standard evaluation set (10-12 prompts) is processed well under the 5-minute requirement. 
* **Encoding Optimization:** The execution speed was drastically improved by encoding the `system_prompt` and `user_prompt` only once before the generation loop. Subsequent generated tokens are simply appended by their `token_id` to the 1D list, avoiding heavy string re-tokenization at every step.
* **Reliability:** The output achieves 100% parseable JSON with complete schema compliance. 

## Challenges Faced
1. **Performance Bottlenecks:** Initially, re-encoding the entire text history at each token generation caused exponential slowdowns. Optimizing the input IDs list manually solved this.
2. **MyPy Strictness:** Ensuring all type hints passed `mypy --strict` alongside standard configurations required explicit generic typing (e.g., `Tuple[str, int]`, `List[str]`) and managing SQLite cache conflicts.
3. **Handling Subwords and Spacing:** Dealing with tokenizers (like BPE) that use `Ġ` for leading spaces required careful text alignment when calculating the remainder of target strings to allow.

## Testing Strategy
The implementation was validated using a custom "stress-test" suite of prompts designed to push the boundaries of the constraints:
* **Syntax Edge Cases:** Empty strings, heavily escaped internal quotes (e.g., `\"hello\"`), and complex regex patterns containing backslashes.
* **Type Forcing:** Testing negative decimals to ensure the `number` mask preserves `-` and `.`.
* **Enum Traps:** Prompting the LLM with invalid enum choices (e.g., "make me a superadmin") to verify the mask forces a fallback to the nearest valid schema option.
* **Graceful Degradation:** Malformed input files and missing data are caught via structured `try/except` blocks, ensuring the main process never crashes unexpectedly.

## Instructions
### Installation
Ensure you have `uv` installed, then synchronize the environment:
```bash
make install