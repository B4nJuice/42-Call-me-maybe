from ..io.io_manager import IOManager
from pydantic import BaseModel
from typing import Any
import importlib.util
import os


class FunctionExecutor(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    io_man: IOManager
    _functions:  = PrivateAttr()


    def model_post_init(self, __context):
        self.

    def execute_function(self, function_name: str, params: dict[str, Any]):
        function_path = self.io_man.args.get("function_path")[0]
        print(os.path.abspath(function_path))
        spec = importlib.util.spec_from_file_location("functions", function_path)
        functions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(functions)
        print(functions.__getattribute__(function_name)(**params))
