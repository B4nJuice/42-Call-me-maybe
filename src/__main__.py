import asyncio
from .utils.log import Colors, colored_text, clear_current_line


def _format_status(status: str, spin_char: str) -> str:
    if status == "running":
        return f"running {spin_char}"
    return status


def _render_prompt_table(
    prompt_texts: list[str],
    statuses: list[str],
    tokens: list[int],
    spin_char: str = "-"
) -> int:
    index_width = max(2, len(str(len(prompt_texts) - 1)))
    status_width = 10
    token_width = 6

    print(colored_text([Colors.BOLD, Colors.CYAN], "Prompt queue"))

    header_id = f"{'ID':<{index_width}}"
    header_status = f"{'STATUS':<{status_width}}"
    header_tokens = f"{'TOKENS':>{token_width}}"
    header_prompt = "PROMPT"

    header_plain = (
        f"{header_id} | {header_status} | {header_tokens} | {header_prompt}"
    )
    header = (
        f"{colored_text([Colors.BOLD], header_id)} | "
        f"{colored_text([Colors.BOLD], header_status)} | "
        f"{colored_text([Colors.BOLD], header_tokens)} | "
        f"{colored_text([Colors.BOLD], header_prompt)}"
    )
    separator = "-" * len(header_plain)

    print(header)
    print(colored_text([Colors.CYAN], separator))

    for idx, text in enumerate(prompt_texts):
        status_name = statuses[idx]
        status = _format_status(status_name, spin_char)
        token_value = "-" if statuses[idx] == "pending" else str(tokens[idx])

        status_colors = {
            "pending": [Colors.YELLOW],
            "running": [Colors.CYAN, Colors.BOLD],
            "done": [Colors.GREEN, Colors.BOLD],
            "error": [Colors.RED, Colors.BOLD],
        }

        colored_id = colored_text([Colors.MAGENTA], f"{idx:<{index_width}}")
        colored_status = colored_text(
            status_colors.get(status_name, [Colors.YELLOW]),
            f"{status:<{status_width}}"
        )
        colored_tokens = colored_text(
            [Colors.YELLOW], f"{token_value:>{token_width}}"
        )
        colored_prompt = colored_text([Colors.CYAN], text)

        print(
            f"{colored_id} | "
            f"{colored_status} | "
            f"{colored_tokens} | "
            f"{colored_prompt}"
        )

    return len(prompt_texts) + 3


def _redraw_prompt_table(
    rendered_lines: int,
    prompt_texts: list[str],
    statuses: list[str],
    tokens: list[int],
    spin_char: str = "-"
) -> int:
    print(f"\x1b[{rendered_lines}F", end="")
    for _ in range(rendered_lines):
        clear_current_line()
        print()
    print(f"\x1b[{rendered_lines}F", end="")
    return _render_prompt_table(prompt_texts, statuses, tokens, spin_char)


async def main() -> None:
    from .model.model import LLMModel
    from .IO.IOManager import IOManager

    io_man: IOManager = IOManager()
    io_man.parse_args()

    llm_model = LLMModel()
    prompts = list(io_man.get_input())
    is_debug = bool(io_man.args.get("debug"))
    prompt_texts = [str(prompt.get("prompt", "")) for prompt in prompts]
    statuses = ["pending"] * len(prompt_texts)
    tokens = [0] * len(prompt_texts)
    responses = [None] * len(prompt_texts)
    errors = [None] * len(prompt_texts)

    rendered_lines = _render_prompt_table(prompt_texts, statuses, tokens)

    for idx, prompt in enumerate(prompts):
        try:
            text = prompt.get("prompt")
            executor = await llm_model.get_prompt_response(text, io_man)

            spinner = "⣾⣽⣻⢿⡿⣟⣯⣷"
            spin_idx = 0
            statuses[idx] = "running"

            while not executor.is_finished:
                if executor.task is not None and executor.task.done():
                    task_error = executor.task.exception()
                    if task_error is not None:
                        raise task_error
                    break

                spin_char = spinner[spin_idx % len(spinner)]
                spin_idx += 1
                tokens[idx] = executor.token

                rendered_lines = _redraw_prompt_table(
                    rendered_lines,
                    prompt_texts,
                    statuses,
                    tokens,
                    spin_char
                )

                await asyncio.sleep(0.1)

            tokens[idx] = executor.token
            statuses[idx] = "done"
            responses[idx] = executor.prompt_response
            rendered_lines = _redraw_prompt_table(
                rendered_lines, prompt_texts, statuses, tokens
            )
        except Exception as e:
            statuses[idx] = "error"
            errors[idx] = str(e)
            rendered_lines = _redraw_prompt_table(
                rendered_lines, prompt_texts, statuses, tokens
            )
            continue

    if is_debug:
        print()
        for idx in range(len(prompt_texts)):
            if errors[idx] is not None:
                print(
                    colored_text(
                        [Colors.RED, Colors.BOLD],
                        f"✘ Error [{idx}]: {errors[idx]}"
                    )
                )
                continue
            print(
                colored_text(
                    [Colors.GREEN], f"✔ Result [{idx}] [{tokens[idx]}]:"
                )
            )
            print(responses[idx])


if __name__ == "__main__":
    # try:
    asyncio.run(main())
    # except Exception as e:
    #     print(e)
