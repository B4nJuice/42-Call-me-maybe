import asyncio
from typing import Any, Protocol, cast
from pydantic import BaseModel, PrivateAttr
import llm_sdk
import numpy as np

from src.io.io_manager import IOManager


class LLMProtocol(Protocol):
    def encode(self, text: str) -> Any:
        ...

    def decode(self, token_id: int) -> str:
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        ...


class LLMModel(BaseModel):
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str | None = None
    _model: LLMProtocol = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        model_cls = getattr(llm_sdk, "Small_LLM_Model")
        self._model = cast(LLMProtocol, model_cls(
            model_name=self.model_name,
            device=self.device
        ))

    @property
    def model(self) -> LLMProtocol:
        return self._model

    async def get_prompt_response(
        self, prompt: str, io_man: IOManager
    ) -> "PromptExecutor":
        prompt_executor: PromptExecutor = PromptExecutor(
            model=self.model,
            io_man=io_man,
            prompt=prompt
        )
        prompt_executor.task = asyncio.create_task(
            asyncio.to_thread(prompt_executor.get_prompt_response)
        )
        return prompt_executor


class PromptExecutor(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: LLMProtocol
    io_man: IOManager
    prompt: str

    token: int = 0
    prompt_response: dict[str, Any] = {}
    function_name: str = ""
    function_params: dict[str, Any] = {}
    function_param_desc: dict[str, Any] = {}
    is_finished: bool = False
    avg_logits: float = 0.0
    task: asyncio.Task[Any] | None = None

    def _get_arg_first(self, key: str) -> Any:
        values: Any = self.io_man.args.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Missing argument: {key}")
        return values[0]

    def get_function_params(self) -> dict[str, str]:
        params: dict[str, Any] = self.function_param_desc.get("parameters", {})
        params_list: str = "\n".join(
            f"{key}:type = {value.get('type')}"
            for key, value in params.items()
            if isinstance(value, dict)
        )
        params_format: str = " ".join(
            f"{key}=\"<value>\"" for key in params.keys()
        )

        parameters_context_path: str | None = self.io_man.config.get(
            "parameters_context"
        )
        if parameters_context_path is None:
            raise ValueError("Missing 'parameters_context' in configuration")

        with open(parameters_context_path) as context:
            rule_context: str = context.read()

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

            param_buffer: str = ""
            while '"' not in param_buffer:
                logits: list[float] = self.model.get_logits_from_input_ids(
                    token_context + token_prompt
                )
                logits_arr = np.asarray(logits)
                next_id: int = int(np.argmax(logits_arr))
                token_prompt.append(next_id)
                next_text: str = self.model.decode(next_id)
                param_buffer += next_text.replace("\n", "")

                self.token += 1
                max_token_value: Any = self._get_arg_first("max_token")
                max_token = int(max_token_value)
                if self.token > max_token:
                    raise ValueError(
                        "Response failed to respond in max-token."
                    )
            result[key] = param_buffer.split('"', maxsplit=1)[0].strip()

        return result

    def get_function_name(self) -> str:
        fd_context: str = self.io_man.get_function_definitions_context()

        name_context_path: str | None = self.io_man.config.get("name_context")
        if name_context_path is None:
            raise ValueError("Missing 'name_context' in configuration")

        with open(name_context_path) as context:
            rule_context: str = context.read()

        rule_context = rule_context.format(fd_context=fd_context)

        token_context: list[int] = self.model.encode(rule_context).tolist()[0]

        token_prompt: list[int] = self.model.encode(
            f"Input= {self.prompt} \nFunction= \""
        ).tolist()[0]
        result: str = ""

        name_logits: list[float] = []

        while not result.strip().endswith('"'):
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
        confidence_value: Any = self._get_arg_first("confidence")
        confidence: float = float(confidence_value)
        if self.avg_logits < confidence:
            raise ValueError(
                f"Confidence ({self.avg_logits:.2f}) is below the threshold."
            )

        return result.replace('"', "").strip()

    def get_prompt_response(self) -> dict[str, Any]:
        self.function_name = self.get_function_name()
        try:
            self.function_param_desc = next(
                fd
                for fd in self.io_man.function_definitions
                if fd.get("name") == self.function_name
            )
        except Exception as exc:
            raise NameError(
                f"Failed to find a function ({self.function_name})"
            ) from exc

        self.function_params = self.get_function_params()
        self.format_params()

        self.prompt_response = {
            "prompt": self.prompt,
            "name": self.function_name,
            "parameters": self.function_params,
        }

        self.is_finished = True

        return self.prompt_response

    def format_params(self) -> None:
        params_schema: dict[str, Any] = self.function_param_desc.get(
            "parameters", {}
        )
        formatted_params: dict[str, str | int | float | bool] = {}

        for key, value in self.function_params.items():
            param_type: str | None = params_schema.get(key, {}).get("type")
            if param_type is None:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            if param_type == "boolean":
                normalized: str = str(value).strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    formatted_params[key] = True
                elif normalized in {"false", "0", "no", "n", "off"}:
                    formatted_params[key] = False
                else:
                    raise ValueError(
                        f"Invalid boolean value for '{key}': {value}"
                    )
            else:
                casters: dict[
                    str, type[str] | type[int] | type[float]
                ] = {
                    "number": float,
                    "integer": int,
                    "string": str,
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

        self.function_params = formatted_params
