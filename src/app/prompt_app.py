import asyncio
from typing import Any
from pydantic import BaseModel, PrivateAttr

from src.io.io_manager import IOManager
from src.model.model import LLMModel
from src.ui.prompt_table import PromptTableRenderer
from src.utils.terminal import Colors, TerminalStyler
from src.utils.function_executor import FunctionExecutor


class PromptApplication(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    _io_manager: IOManager = PrivateAttr()
    _llm_model: LLMModel = PrivateAttr()
    _function_executor: FunctionExecutor = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._io_manager = IOManager()

        model_values: Any = self.io_manager.args.get("model")
        device_values: Any = self.io_manager.args.get("device")
        if (
            not isinstance(model_values, list)
            or not model_values
            or not isinstance(model_values[0], str)
        ):
            raise ValueError("Missing or invalid model argument")

        device: str | None = None
        if isinstance(device_values, list) and device_values:
            first_device: Any = device_values[0]
            if first_device is None or isinstance(first_device, str):
                device = first_device

        self._llm_model = LLMModel(
            model_name=model_values[0],
            device=device,
        )
        self._function_executor = FunctionExecutor(io_man=self.io_manager)

    @property
    def io_manager(self) -> IOManager:
        return self._io_manager

    @property
    def llm_model(self) -> LLMModel:
        return self._llm_model

    @property
    def function_executor(self) -> FunctionExecutor:
        return self._function_executor

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
        returns: list[Any] = [{}] * len(prompt_texts)

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
                current_response: dict[str, Any] = executor.prompt_response
                responses[idx] = current_response

                if execute_function:
                    returns[idx] = self.function_executor.execute_function(
                        function_name=executor.function_name,
                        params=executor.function_params
                    )
                    if returns[idx]:
                        current_response.update(
                            {"return": returns[idx].get("return")}
                        )
                        current_response.update(
                            {"output": returns[idx].get("output")}
                        )

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
