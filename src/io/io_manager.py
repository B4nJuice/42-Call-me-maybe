import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
                        BaseModel,
                        ConfigDict,
                        PrivateAttr,
                        TypeAdapter,
                        field_validator
                    )


class InputItem(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("prompt must be a non-empty string")
        return v


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["number", "string", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class IOManager(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _args_config: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _args: dict[str, Any] = PrivateAttr(default_factory=dict)
    _type_map: dict[str, type[Any]] = PrivateAttr(
        default_factory=lambda: {
            "integer": int,
            "float": float,
            "string": str,
            "boolean": bool,
        }
    )
    _config: dict[str, str] = PrivateAttr(default_factory=dict)
    _input: list[dict[str, str]] = PrivateAttr(default_factory=list)
    _function_definitions: list[dict[str, Any]] = PrivateAttr(
            default_factory=list
        )

    def model_post_init(self, __context: Any) -> None:
        self.parse_args()
        self.get_config()
        self.get_input()
        self.get_function_definitions()

    @property
    def args_config(self) -> dict[str, dict[str, Any]]:
        return self._args_config

    @property
    def args(self) -> dict[str, Any]:
        return self._args

    @property
    def type_map(self) -> dict[str, type[Any]]:
        return self._type_map

    @property
    def config(self) -> dict[str, str]:
        return self._config

    @property
    def input(self) -> list[dict[str, str]]:
        return self._input

    @property
    def function_definitions(self) -> list[dict[str, Any]]:
        return self._function_definitions

    def parse_args(self) -> dict[str, Any]:
        args_config: dict[str, dict[str, Any]] = {}
        parser = argparse.ArgumentParser()

        args_path: Path = Path(__file__).with_name("args.json")
        with open(args_path) as args_file:
            args_config = json.load(args_file)

        for key, data in args_config.items():
            alias: str | None = data.get("alias")
            type_name: Any = data.get("type")
            arg_type: type[Any] | None = (
                self.type_map.get(type_name)
                if isinstance(type_name, str)
                else None
            )
            action: str | None = data.get("action")
            required: bool = data.get("required") == 1
            nargs: Any = data.get("nargs")

            arg_kwargs: dict[str, Any] = {
                "action": action,
                "required": required,
            }

            if action in ["store_true", "store_false"]:
                arg_kwargs["default"] = bool(data.get("default", False))
            else:
                default: Any = None
                if not required and arg_type is not None:
                    default = arg_type(data.get("default"))
                arg_kwargs["type"] = arg_type
                arg_kwargs["default"] = [default]
                arg_kwargs["nargs"] = nargs

            if alias is None:
                parser.add_argument(key, **arg_kwargs)
            else:
                parser.add_argument(key, alias, **arg_kwargs)

        self._args_config = args_config
        self._args = vars(parser.parse_args())
        return self._args

    def get_config(self) -> dict[str, str]:
        with open("./src/config/config.json") as config:
            self._config = json.load(config)

        return self._config

    def store_in_output(self, data: Any, mode: str = "w+") -> None:
        output_values: Any = self.args.get("output")
        if not isinstance(output_values, list) or not output_values:
            raise ValueError("Missing output path in arguments")
        output_path_str: Any = output_values[0]
        if not isinstance(output_path_str, str):
            raise ValueError("Output path must be a string")

        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, mode) as output_file:
            json.dump(data, output_file, indent="\t", ensure_ascii=False)
            output_file.write("\n")

    def get_input(self) -> list[dict[str, str]]:
        input_values: Any = self.args.get("input")
        if not isinstance(input_values, list) or not input_values:
            raise ValueError("Missing input path in arguments")
        input_path: Any = input_values[0]
        if not isinstance(input_path, str):
            raise ValueError("Input path must be a string")

        with open(input_path) as input_file:
            data = json.load(input_file)

        adapter = TypeAdapter(list[InputItem])
        adapter.validate_python(data)

        self._input = data
        return self._input

    def get_function_definitions(self) -> list[dict[str, Any]]:
        fd_values: Any = self.args.get("function_definitions")
        if not isinstance(fd_values, list) or not fd_values:
            raise ValueError(
                "Missing function definitions path in arguments"
            )
        fd_path: Any = fd_values[0]
        if not isinstance(fd_path, str):
            raise ValueError("Function definitions path must be a string")

        with open(fd_path) as fd_file:
            data = json.load(fd_file)

        adapter = TypeAdapter(list[FunctionDefinition])
        adapter.validate_python(data)

        self._function_definitions = data
        return self._function_definitions

    def get_function_definitions_context(self) -> str:
        context: list[str] = []
        for function in self.function_definitions:
            name = function.get("name")
            description = function.get("description")

            params = function.get("parameters", {})
            params_str = ", ".join(
                f"{k}: {v.get('type')}" for k, v in params.items()
            )

            returns = function.get("returns", {}).get("type")

            function_str = (
                f"function_name= {name}\n"
                f"description= {description}\n"
                f"parameters= {params_str}\n"
                f"returns= {returns}\n"
            )

            context.append(function_str)

        return "\n".join(context)
