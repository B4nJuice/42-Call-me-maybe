from ..IO.IOManager import IOManager
from typing import Any
import llm_sdk
import asyncio


class LLMModel:
    def __init__(self):
        self.model: llm_sdk.Small_LLM_Model =\
            llm_sdk.Small_LLM_Model(device="cpu")

    async def get_prompt_response(self, prompt: str, io_man: IOManager) -> str:
        prompt_executor: PromptExecutor = PromptExecutor(
            self.model, io_man, prompt
        )
        prompt_executor.task = asyncio.create_task(
            asyncio.to_thread(
                prompt_executor.get_prompt_response
                )
        )
        return prompt_executor


class PromptExecutor:
    def __init__(
                    self,
                    model: llm_sdk.Small_LLM_Model,
                    io_man: IOManager,
                    prompt: str
                ):

        self.token: int = 0
        self.model: llm_sdk.Small_LLM_Model = model
        self.io_man: IOManager = io_man
        self.prompt: str = prompt
        self.prompt_response: str = ""
        self.function_name: str = ""
        self.function_param: dict[str, str] = {}
        self.function_param_desc: dict[str, Any] = {}
        self.is_finished: bool = False
        self.task: asyncio.Task[Any] | None = None

    def get_function_params(self) -> dict[str, str]:
        params = self.function_param_desc.get("parameters", {})
        params_list = "\n".join(
            f"{key}:type = {params.get(key).get('type')}"
            for key in params.keys()
        )
        params_format = " ".join(f"{key}=\"<value>\"" for key in params.keys())

        rule_context: str = f"""Parameter finder
Rules=
- Only output function call
- Extract intact parameters from user input
- Never execute, call or simulate
- No explanation

Function= {self.function_name}
parameters list=
{params_list}

Format=
Function= {self.function_name}
Parameters= {params_format}
"""

        token_context: list[int] = self.model.encode(rule_context).tolist()[0]

        token_prompt: list[int] = self.model.encode(
                f"Input= {self.prompt} \nFunction= \"{self.function_name}\"\n"
            ).tolist()[0]

        result: dict[str, str] = {}

        for key in params.keys():
            token_key: list[int] = self.model.encode(
                f"{key}=\""
            ).tolist()[0]
            token_prompt += token_key

            param_buffer = ""
            while '"' not in param_buffer:
                logits: list[float] = self.model.get_logits_from_input_ids(
                        token_context + token_prompt
                    )

                next_id = logits.index(max(logits))
                token_prompt.append(next_id)
                next_text: str = self.model.decode(next_id)
                param_buffer += next_text.replace("\n", "")

                self.token += 1
            result[key] = param_buffer.split('"', maxsplit=1)[0]

        return result

    def get_function_name(self) -> str:
        fd_context: str = self.io_man.get_function_definitions_context()

        rule_context: str = f"""
Function selector.

Rules=
- Only output function call
- No explanation
- Valid regex as string with ""

Functions=
{fd_context}

Format=
Function= "<name>"
Parameters= <key>="<value>" <key2>="<value2>"
"""

        token_context: list[int] = self.model.encode(rule_context).tolist()[0]

        token_prompt: list[int] = self.model.encode(
                f"Input= {self.prompt} \nFunction= \""
            ).tolist()[0]
        result: str = ""

        while not result.strip().endswith("\""):
            logits: list[float] = self.model.get_logits_from_input_ids(
                    token_context + token_prompt
                )
            next_id = logits.index(max(logits))
            token_prompt.append(next_id)
            next_text: str = self.model.decode(next_id)
            result += next_text

            self.token += 1

        return result.replace("\"", "").strip()

    def get_prompt_response(self) -> dict[str, Any]:

        self.function_name = self.get_function_name()
        try:
            self.function_param_desc = list(
                filter(
                    lambda d: d.get("name") == self.function_name,
                    self.io_man.get_function_definitions()
                    )
                )[0]
        except Exception:
            raise NameError(
                f"Failed to find a function ({self.function_name})"
            )

        self.function_param = self.get_function_params()

        self.prompt_response = {
            "prompt": self.prompt,
            "name": self.function_name,
            "parameters": self.function_param
        }

        return self.prompt_response

    def format_params(self) -> None:
        #transformer les params dans leur bon type
