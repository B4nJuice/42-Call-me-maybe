import json
from typing import Any


class IOManager:
    def __init__(self):
        self.args_config: dict[str, str] = {}

    def parse_args(self) -> dict[str, str]:
        import sys

        args_config: dict[str, str] = {}

        with open("./src/IO/args.json") as args_file:
            json_data: str = args_file.read()
            args_config = json.loads(json_data)

        argv: list[str] = sys.argv[1:]
        argc: int = len(argv)

        args_config_keys: list[str] = list(args_config.keys())

        key: str = ""
        for idx, arg in enumerate(argv):
            if not idx % 2:
                if arg not in args_config_keys:
                    raise ValueError(f"{arg}: unknown parameter.")
                if idx == argc - 1:
                    raise ValueError(f"{arg}: no value gived.")
                key = arg
            else:
                args_config.update({key: arg})

        self.args_config = args_config

    def store_in_output(self, data: Any, mode: str = "w") -> None:
        with open(self.args_config.get("--output"), mode) as output_file:
            json.dump(data, output_file)

    def get_input(self) -> list[dict[Any]]:
        input_data: list[dict[Any]] = []
        with open(self.args_config.get("--input")) as input_file:
            input_data = json.loads(input_file.read())
        for idx, prompt in enumerate(input_data):

            if not isinstance(prompt.get("prompt"), str):
                raise ValueError(f"Prompt {idx}: must contain a valid string.")
            if len(prompt) != 1:
                raise ValueError(f"Prompt {idx}: contains multiple fields.")

        return input_data

    def get_function_definitions(self) -> list[dict[Any]]:
        with open(self.args_config.get("--function_definitions")) as fd_file:
            return json.loads(fd_file.read())
