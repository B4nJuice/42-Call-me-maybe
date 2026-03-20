import asyncio
from enum import Enum


class colors(Enum):
    RESET = "\033[0m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"


async def main() -> None:
    from .model.model import LLMModel
    from .IO.IOManager import IOManager

    io_man: IOManager = IOManager()
    io_man.parse_args()

    llm_model = LLMModel()

    functions_context = io_man.get_function_definitions_context()

    context = f"""
Function selector.

Rules=
- Only output function call
- No explanation
- End with |END|
- Valid regex as string with \"\"
- Only use =

Functions=
{functions_context}

Format=
Function= <name>
Parameters= <key>=\"<value>\", <key2>=\"<value2>\"|END|
"""

    for idx, prompt in enumerate(io_man.get_input()):
        text = prompt.get("prompt")

        print(f"{colors.BOLD.value}{colors.CYAN.value}▶ Prompt [{idx}]:{colors.RESET.value} {text}")

        executor = await llm_model.get_prompt_response(text, context)

        spinner = "⣾⣽⣻⢿⡿⣟⣯⣷"
        spin_idx = 0

        while executor.prompt_response == "":
            spin_char = spinner[spin_idx % len(spinner)]
            spin_idx += 1

            # print(
            #     f"\r{colors.CYAN.value}Genrating [{idx}] {spin_char} "
            #     f"{colors.MAGENTA.value}{executor.token} tokens{colors.RESET.value}",
            #     end="",
            #     flush=True
            # )

            await asyncio.sleep(0.1)

        print("\r" + " " * 80, end="\r")

        print(f"{colors.GREEN.value}✔ Result [{idx}] [{executor.token}]:{colors.RESET.value}")
        print(f"{executor.prompt_response.strip()}\n")


if __name__ == "__main__":
    # try:
    asyncio.run(main())
    # except Exception as e:
    #     print(e)
