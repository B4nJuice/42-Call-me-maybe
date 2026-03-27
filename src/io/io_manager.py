import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator


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


class IOManager:
    def __init__(self) -> None:
        self.args_config: dict[str, dict[str, Any]] = {}
        self.args: dict[str, Any] = {}
        self.type_map: dict[str, type[Any]] = {
            "integer": int,
            "float": float,
            "string": str,
            "boolean": bool,
        }
        self.config: dict[str, str] = {}
        self.input: list[dict[str, str]] = []
        self.function_definitions: list[dict[str, Any]] = []
        self.parse_args()
        self.get_config()
        self.get_input()
        self.get_function_definitions()

    def parse_args(self) -> dict[str, Any]:
        args_config: dict[str, dict[str, Any]] = {}
        parser = argparse.ArgumentParser()

        args_path: Path = Path(__file__).with_name("args.json")
        with open(args_path) as args_file:
            args_config = json.load(args_file)

        for key, data in args_config.items():
            alias: str | None = data.get("alias")
            arg_type: type[Any] | None = self.type_map.get(data.get("type"))
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

        self.args_config = args_config
        self.args = vars(parser.parse_args())
        return self.args

    def get_config(self) -> dict[str, str]:
        with open("./src/config/config.json") as config:
            self.config = json.load(config)

        return self.config

    def store_in_output(self, data: Any, mode: str = "w+") -> None:
        output_path_str: str | None = self.args.get("output")[0]
        if output_path_str is None:
            raise ValueError("Missing output path in arguments")

        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, mode) as output_file:
            json.dump(data, output_file, indent="\t", ensure_ascii=False)
            output_file.write("\n")

    def get_input(self) -> list[dict[str, str]]:
        with open(self.args.get("input")[0]) as input_file:
            data = json.load(input_file)

        adapter = TypeAdapter(list[InputItem])
        adapter.validate_python(data)

        self.input = data
        return self.input

    def get_function_definitions(self) -> list[dict[str, Any]]:
        with open(self.args.get("function_definitions")[0]) as fd_file:
            data = json.load(fd_file)

        adapter = TypeAdapter(list[FunctionDefinition])
        adapter.validate_python(data)

        self.function_definitions = data
        return self.function_definitions

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
