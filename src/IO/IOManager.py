import json
import sys
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
        self.args_config: dict[str, str] = {}

    def parse_args(self) -> dict[str, str]:
        args_config: dict[str, str] = {}

        with open("./src/IO/args.json") as args_file:
            args_config = json.load(args_file)

        argv: list[str] = sys.argv[1:]
        argc: int = len(argv)

        args_config_keys = list(args_config.keys())

        key = ""
        for idx, arg in enumerate(argv):
            if idx % 2 == 0:
                if arg not in args_config_keys:
                    raise ValueError(f"{arg}: unknown parameter.")

                if idx == argc - 1:
                    raise ValueError(f"{arg}: no value given.")

                key = arg
            else:
                args_config[key] = arg

        self.args_config = args_config
        return args_config

    def store_in_output(self, data: Any, mode: str = "w") -> None:
        with open(self.args_config.get("--output"), mode) as output_file:
            json.dump(data, output_file)

    def get_input(self) -> list[dict[str, str]]:
        with open(self.args_config.get("--input")) as input_file:
            data = json.load(input_file)

        adapter = TypeAdapter(list[InputItem])
        adapter.validate_python(data)

        return data

    def get_function_definitions(self) -> list[dict[str, Any]]:
        with open(self.args_config.get("--function_definitions")) as fd_file:
            data = json.load(fd_file)

        adapter = TypeAdapter(list[FunctionDefinition])
        adapter.validate_python(data)

        return data
