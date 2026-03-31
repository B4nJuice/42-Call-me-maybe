from ..io.io_manager import IOManager
from pydantic import BaseModel, PrivateAttr
from typing import Any, types
import contextlib
import importlib.util
import io
import os


class FunctionExecutor(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    io_man: IOManager
    _functions: types.ModuleType = PrivateAttr()

    def model_post_init(self, __context):
        function_path: str = self.io_man.args.get("function_path")[0]
        spec = importlib.util.spec_from_file_location("functions", function_path)
        self._functions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.functions)

    @property
    def functions(self):
        return self._functions

    def execute_function(
                self,
                function_name: str,
                params: dict[str, Any]
            ) -> Any:
        try:
            output: io.StringIO = io.StringIO()
            
            function: callable = self.functions.__getattribute__(function_name)
            with contextlib.redirect_stdout(output):
                return {
                    "function_name": function_name,
                    "params": params,
                    "return": function(**params),
                    "output": output.getvalue()
                    }
        except Exception:
            return
