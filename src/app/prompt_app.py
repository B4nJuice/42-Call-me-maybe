import asyncio
from typing import Any

from ..io.io_manager import IOManager
from ..model.model import LLMModel
from ..ui.prompt_table import PromptTableRenderer
from ..utils.terminal import Colors, TerminalStyler
from ..utils.function_executor import FunctionExecutor


class PromptApplication:
    def __init__(self) -> None:
        self.io_manager: IOManager = IOManager()
        self.io_manager.parse_args()
        self.llm_model: LLMModel = LLMModel(
                model_name=self.io_manager.args.get("model")[0],
                device=self.io_manager.args.get("device")[0]
            )
        self.function_executor: FunctionExecutor = (
                FunctionExecutor(io_man=self.io_manager)
            )


    async def run(self) -> None:
        prompts: list[dict[str, str]] = list(self.io_manager.get_input())
        is_debug: bool = bool(self.io_manager.args.get("debug"))
        is_no_output: bool = bool(self.io_manager.args.get("no_output"))
        execute_function: bool = bool(
                self.io_manager.args.get("execute_functions")
            )

        prompt_texts: list[str] = [
            str(prompt.get("prompt", "")) for prompt in prompts
        ]
        responses: list[dict[str, Any] | None] = [None] * len(prompt_texts)
        errors: list[str | None] = [None] * len(prompt_texts)
        returns: list[Any] = [None] * len(prompt_texts)

        table_renderer: PromptTableRenderer = PromptTableRenderer(
                prompt_texts=prompt_texts
            )
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

                if execute_function:
                    try:
                        returns[idx] = self.function_executor.execute_function(
                            function_name=executor.function_name,
                            params=executor.function_params
                        )
                        responses[idx].update(
                            {"return": returns[idx].get("return")}
                        )
                        responses[idx].update(
                            {"output": returns[idx].get("output")}
                        )
                    except Exception as e:
                        returns[idx] = e

                self.io_manager.store_in_output(
                    [
                        response
                        for response in responses
                        if response is not None
                    ]
                )

                if not is_no_output:
                    table_renderer.redraw()
            except Exception as e:
                table_renderer.set_status(idx, "error")
                errors[idx] = str(e)
                responses[idx] = {
                    "prompt": prompt_texts[idx],
                    "error": str(e),
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

        if execute_function and not is_no_output:
            table_renderer.render_returns(returns)

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
