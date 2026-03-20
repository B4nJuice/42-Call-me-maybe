import json
import argparse
from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter, field_validator, ConfigDict


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

    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class IOManager:
    def __init__(self):
        self.args_config: dict[str, dict[str, Any]] = {}
        self.args: dict[str, Any] = {}
        self.type_map: dict[str, type] = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }

    def parse_args(self) -> dict[str, str]:
        args_config: dict[str, str] = {}
        parser = argparse.ArgumentParser()

        with open("./src/IO/args.json") as args_file:
            args_config = json.load(args_file)

        for key, data in args_config.items():
            alias: str = data.get("alias")
            arg_type: type = self.type_map.get(data.get("type"))
            required: bool = data.get("required") == 1
            default: Any = None if required else arg_type(data.get("default"))
            action: str = data.get("action")
            parser.add_argument(
                    key,
                    alias,
                    type=arg_type,
                    required=required,
                    default=default,
                    action=action
                )

        self.args_config = args_config
        self.args = vars(parser.parse_args())
        return self.args

    def store_in_output(self, data: Any, mode: str = "w") -> None:
        with open(self.args_config.get("--output"), mode) as output_file:
            json.dump(data, output_file)

    def get_input(self) -> list[dict[str, str]]:
        with open(self.args.get("input")) as input_file:
            data = json.load(input_file)

        adapter = TypeAdapter(list[InputItem])
        adapter.validate_python(data)

        return data

    def get_function_definitions(self) -> list[dict[str, Any]]:
        with open(self.args.get("function_definitions")) as fd_file:
            data = json.load(fd_file)

        adapter = TypeAdapter(list[FunctionDefinition])
        adapter.validate_python(data)

        return data
