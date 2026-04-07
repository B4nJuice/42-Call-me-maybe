*This project has been created as part of the 42 curriculum by lgirard.*

# Call-Me-Maybe

## Description
Call-Me-Maybe is an AI project built with the `llm_sdk` library. Its goal is to create a function-selector agent that chooses a function from a user question and a list of structured function definitions. It outputs the selected function in valid JSON format with constrained decoding.

## Instructions

### Makefile targets

- `make install`: creates the environment, builds the `llm_sdk` library, and installs dependencies.
- `make run`: runs the application (`python -m src`).
- `make lint`: runs `flake8` + `mypy` (custom config).
- `make lint-strict`: runs `flake8` + `mypy --strict`.
- `make clean`: removes Python caches, `llm_sdk` dist files, and output data.
- `make fclean`: runs `clean` + removes `.venv` and library artifacts.
- `make remove-cache`: removes the `uv` cache (`/tmp/.uv-cache`).

### Run the application

Standard command:

- `make run`

Pass CLI options through the `ARGS` variable:

- `make run ARGS="--model Qwen/Qwen3-0.6B --device cpu --debug"`

### Available flags

The following flags are defined in [src/io/args.json](src/io/args.json):

- `--input`, `-i`: path to the prompts input file.
  - default: `./data/input/function_calling_tests.json`
- `--function_definitions`, `-fd`: path to the function definitions file.
  - default: `./data/input/function_definitions.json`
- `--output`, `-o`: path to the JSON output file.
  - default: `./data/output/function_calls.json`
- `--model`, `-m`: LLM model name.
  - default: `Qwen/Qwen3-0.6B`
- `--device`, `-dv`: inference device.
  - default: `cpu`
- `--function-path`, `-fp`: Python file containing executable functions.
  - default: `./src/default_functions/default_functions.py`
- `--confidence`, `-c`: minimum confidence threshold.
  - default: `22`
- `--max-token`, `-mt`: maximum number of generated tokens.
  - default: `60`
- `--debug`, `-d`: enables debug output.
  - default: `false`
- `--no-output`, `-no`: disables terminal rendering.
  - default: `false`
- `--execute-functions`, `-ef`: executes the predicted function with generated parameters.
  - default: `false`

## Algorithm explanation

The project uses a constrained decoding workflow in two stages:

1. **Function name selection**
   - The model receives a context prompt listing all available functions (name, description, parameters, return type).
   - Decoding starts from a fixed pattern (`Function= "`) and continues token by token.
   - At each step, the next token is selected with greedy decoding (`argmax` on logits).
   - Generation stops only when the closing quote is produced, which constrains output to a single function-name string.

2. **Parameter generation**
   - Once the function is selected, the model receives a parameter-specific template that includes expected keys and types.
   - Each parameter is decoded independently with a constrained prefix (`key="`).
   - Decoding for each field stops at the next quote, enforcing string-bounded values.
   - Values are then validated and cast to schema types (`integer`, `number`, `boolean`, `string`).

Additional controls:
- A **max-token** threshold limits runaway decoding.
- A **confidence** threshold (based on average max logits on function-name decoding) rejects uncertain predictions.
- Input and function schemas are validated with Pydantic before execution.

## Design decisions

- **Schema-first design**: function definitions and input payloads are validated early with Pydantic models.
- **Two-pass decoding**: function name first, then parameters, to reduce ambiguity and simplify validation.
- **Greedy decoding**: deterministic and easy to debug for constrained outputs.
- **Separation of concerns**:
  - `IOManager` handles arguments, config, and files.
  - `LLMModel` / `PromptExecutor` handle inference and decoding logic.
  - `FunctionExecutor` isolates dynamic function loading/execution.
  - `PromptTableRenderer` handles terminal UX only.
- **Optional execution mode**: prediction and execution are decoupled via `--execute-functions`.

## Performance analysis

- **Accuracy**
  - Strongly depends on prompt templates and function-definition quality.
  - Constrained output format improves structural correctness (valid keys/types) compared with free-form generation.

- **Speed**
  - Token generation is iterative and synchronous per prompt.
  - The dominant cost is repeated logits computation during constrained decoding.
  - UI refresh and optional function execution add small overhead compared with inference.

- **Reliability**
  - Pydantic validation, explicit type casting, and confidence/token thresholds improve robustness.
  - Errors are captured per prompt and serialized to output, limiting batch-level failures.

## Challenges faced

- **Know the end of the answer**
  - Solved by hard stop conditions and max-tokens limit.

- **Ambiguous function choice**
  - Basically solved by adding a confidence threshold.

## Testing strategy

- **Static analysis**: `make lint` runs `flake8` and `mypy` to catch style/type issues.
- **Runtime validation**:
  - Pydantic validation for input/function schema integrity.
  - Per-prompt exception handling to verify failure isolation and output serialization.
- **Manual scenario testing** with different flags (`--debug`, `--no-output`, `--execute-functions`, custom paths).

## Terminal output example

![Terminal example](https://media.discordapp.net/attachments/860546628564942878/1491002078049468456/image.png?ex=69d61b64&is=69d4c9e4&hm=73bf8b3b4475d276f6a2bf3e3c46a2d7855b4c67589878d5f1a2ea1da58fe93c&=&format=webp&quality=lossless)

## Example usage

- Basic run:
  - `make run`

- Run with debug mode:
  - `make run ARGS="--debug"`

- Run with custom input/output files:
  - `make run ARGS="--input ./data/input/function_calling_tests.json --output ./data/output/function_calls.json"`

- Run with function execution enabled:
  - `make run ARGS="--execute-functions --function-path ./src/default_functions/default_functions.py"`

- Quiet mode (no terminal table rendering):
  - `make run ARGS="--no-output"`

## Resources

- Available models: [Hugging Face](https://huggingface.co/)

- Pydantic documentation: [Pydantic docs](https://docs.pydantic.dev/latest/)

- Common Python documentation: [W3School](https://www.w3schools.com/python/default.asp) | [GFG](https://www.geeksforgeeks.org/python/python-programming-language-tutorial/) | [Official Python docs](https://docs.python.org/3/)