import asyncio
from typing import Any

import llm_sdk
import numpy as np

from ..IO.IOManager import IOManager

PromptResponse = dict[str, Any]


class LLMModel:
    def __init__(self) -> None:
        self.model: llm_sdk.Small_LLM_Model = llm_sdk.Small_LLM_Model()

    async def get_prompt_response(
        self, prompt: str, io_man: IOManager
    ) -> "PromptExecutor":
        prompt_executor: PromptExecutor = PromptExecutor(
            self.model, io_man, prompt
        )
        prompt_executor.task = asyncio.create_task(
            asyncio.to_thread(prompt_executor.get_prompt_response)
        )
        return prompt_executor


class PromptExecutor:
    def __init__(
        self,
        model: llm_sdk.Small_LLM_Model,
        io_man: IOManager,
        prompt: str,
    ) -> None:
        self.token: int = 0
        self.model: llm_sdk.Small_LLM_Model = model
        self.io_man: IOManager = io_man
        self.prompt: str = prompt
        self.prompt_response: PromptResponse = {}
        self.function_name: str = ""
        self.function_param: dict[str, Any] = {}
        self.function_param_desc: dict[str, Any] = {}
        self.is_finished: bool = False
        self.avg_logits: int = 0
        self.task: asyncio.Task[Any] | None = None

    def get_function_params(self) -> dict[str, str]:
        params: dict[str, Any] = self.function_param_desc.get("parameters", {})
        params_list = "\n".join(
            f"{key}:type = {params.get(key).get('type')}"
            for key in params.keys()
        )
        params_format = " ".join(f"{key}=\"<value>\"" for key in params.keys())

        rule_context: str = ""

        with open(self.io_man.config.get("parameters_context")) as context:
            rule_context = context.read()

        rule_context = rule_context.format(
            function_name=self.function_name,
            params_list=params_list,
            params_format=params_format,
        )

        token_context: list[int] = self.model.encode(rule_context).tolist()[0]

        token_prompt: list[int] = self.model.encode(
            f"Input= {self.prompt} \nFunction= \"{self.function_name}\"\n"
        ).tolist()[0]

        result: dict[str, str] = {}

        for key in params.keys():
            token_key: list[int] = self.model.encode(f"{key}=\"").tolist()[0]
            token_prompt += token_key

            param_buffer = ""
            while '"' not in param_buffer:
                logits: list[float] = self.model.get_logits_from_input_ids(
                    token_context + token_prompt
                )
                logits_arr = np.asarray(logits)
                next_id = int(np.argmax(logits_arr))
                token_prompt.append(next_id)
                next_text: str = self.model.decode(next_id)
                param_buffer += next_text.replace("\n", "")

                self.token += 1
            result[key] = param_buffer.split('"', maxsplit=1)[0].strip()

        return result

    def get_function_name(self) -> str:
        fd_context: str = self.io_man.get_function_definitions_context()

        rule_context: str = ""

        with open(self.io_man.config.get("name_context")) as context:
            rule_context = context.read()

        rule_context = rule_context.format(fd_context=fd_context)

        token_context: list[int] = self.model.encode(rule_context).tolist()[0]

        token_prompt: list[int] = self.model.encode(
            f"Input= {self.prompt} \nFunction= \""
        ).tolist()[0]
        result: str = ""

        name_logits: list[float] = []

        while not result.strip().endswith("\""):
            logits: list[float] = self.model.get_logits_from_input_ids(
                token_context + token_prompt
            )
            logits_arr = np.asarray(logits)
            max_logit: float = float(np.max(logits_arr))
            name_logits.append(max_logit)
            next_id: int = int(np.argmax(logits_arr))
            token_prompt.append(next_id)
            next_text: str = self.model.decode(next_id)
            result += next_text

            self.token += 1

        self.avg_logits = sum(name_logits) / self.token
        if self.avg_logits < 24:
            raise ValueError(
                f"Confidence ({self.avg_logits:.2f}) is below the threshold."
            )

        return result.replace("\"", "").strip()

    def get_prompt_response(self) -> PromptResponse:
        self.function_name = self.get_function_name()
        try:
            self.function_param_desc = list(
                filter(
                    lambda d: d.get("name") == self.function_name,
                    self.io_man.function_definitions,
                )
                )[0]
        except Exception:
            raise NameError(
                f"Failed to find a function ({self.function_name})"
            )

        self.function_param = self.get_function_params()
        self.format_params()

        self.prompt_response = {
            "prompt": self.prompt,
            "name": self.function_name,
            "parameters": self.function_param,
        }

        self.is_finished = True

        return self.prompt_response

    def format_params(self) -> None:
        params_schema: dict[str, Any] = self.function_param_desc.get(
            "parameters", {}
        )
        formatted_params: dict[str, Any] = {}

        for key, value in self.function_param.items():
            param_type: str | None = params_schema.get(key, {}).get("type")
            if param_type is None:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            if param_type == "boolean":
                normalized = str(value).strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    formatted_params[key] = True
                elif normalized in {"false", "0", "no", "n", "off"}:
                    formatted_params[key] = False
                else:
                    raise ValueError(
                        f"Invalid boolean value for '{key}': {value}"
                    )
            else:
                casters: dict[str, Any] = {
                    "number": float,
                    "integer": int,
                    "string": str,
                    "boolean": bool,
                }
                caster = casters.get(param_type)

                if caster is None:
                    raise ValueError(
                        f"Unsupported parameter type: {param_type}"
                    )

                try:
                    formatted_params[key] = caster(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid value for '{key}' ({param_type}): {value}"
                    ) from exc

        self.function_param = formatted_params
