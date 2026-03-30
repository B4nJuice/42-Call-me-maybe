import asyncio
from typing import Any

from ..io.io_manager import IOManager
from ..model.model import LLMModel
from ..ui.prompt_table import PromptTableRenderer
from ..utils.terminal import Colors, TerminalStyler


class PromptApplication:
    def __init__(self) -> None:
        self.io_manager: IOManager = IOManager()
        self.llm_model: LLMModel = LLMModel()

    async def run(self) -> None:
        self.io_manager.parse_args()

        prompts: list[dict[str, str]] = list(self.io_manager.get_input())
        is_debug: bool = bool(self.io_manager.args.get("debug"))
        is_no_output: bool = bool(self.io_manager.args.get("no_output"))

        prompt_texts: list[str] = [
            str(prompt.get("prompt", "")) for prompt in prompts
        ]
        responses: list[dict[str, Any] | None] = [None] * len(prompt_texts)
        errors: list[str | None] = [None] * len(prompt_texts)

        table_renderer: PromptTableRenderer = PromptTableRenderer(prompt_texts)
        if not is_no_output:
            table_renderer.render()

        for idx, prompt in enumerate(prompts):
            try:
                prompt_text: str = prompt.get("prompt", "")
                executor = await self.llm_model.get_prompt_response(
                    prompt_text, self.io_manager
                )

                spinner: str = "⣾⣽⣻⢿⡿⣟⣯⣷"
                spin_idx: int = 0
                table_renderer.set_status(idx, "running")

                while not executor.is_finished:
                    if executor.task is not None and executor.task.done():
                        task_error = executor.task.exception()
                        if task_error is not None:
                            raise task_error
                        break

                    spin_char: str = spinner[spin_idx % len(spinner)]
                    spin_idx += 1
                    table_renderer.set_token(idx, executor.token)

                    if not is_no_output:
                        table_renderer.redraw(spin_char)

                    await asyncio.sleep(0.1)

                table_renderer.set_token(idx, executor.token)
                table_renderer.set_status(idx, "done")
                responses[idx] = executor.prompt_response
                self.io_manager.store_in_output(
                    [
                        response
                        for response in responses
                        if response is not None
                    ]
                )

                if not is_no_output:
                    table_renderer.redraw()
            except Exception as exc:
                table_renderer.set_status(idx, "error")
                errors[idx] = str(exc)
                responses[idx] = {
                    "prompt": prompt_texts[idx],
                    "error": str(exc),
                }
                self.io_manager.store_in_output(
                    [
                        response
                        for response in responses
                        if response is not None
                    ]
                )
                if not is_no_output:
                    table_renderer.redraw()
                continue

        if is_debug and not is_no_output:
            self._print_debug_results(
                errors=errors,
                responses=responses,
                tokens=table_renderer.tokens,
            )

    def _print_debug_results(
        self,
        errors: list[str | None],
        responses: list[dict[str, Any] | None],
        tokens: list[int],
    ) -> None:
        print()
        for idx in range(len(tokens)):
            if errors[idx] is not None:
                print(
                    TerminalStyler.colored_text(
                        [Colors.RED, Colors.BOLD],
                        f"✘ Error [{idx}]: {errors[idx]}",
                    )
                )
                continue

            print(
                TerminalStyler.colored_text(
                    [Colors.GREEN],
                    f"✔ Result [{idx}] [{tokens[idx]}]:",
                )
            )
            print(responses[idx])
