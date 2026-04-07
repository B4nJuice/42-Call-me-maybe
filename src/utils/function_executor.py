from src.io.io_manager import IOManager
from pydantic import BaseModel, PrivateAttr
from typing import Any, Callable, cast
import types
import contextlib
import importlib.util
import importlib.abc
import io


class FunctionExecutor(BaseModel):
    """Load and execute user-defined functions from a Python module."""

    model_config = {"arbitrary_types_allowed": True}

    io_man: IOManager
    _functions: types.ModuleType = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Load the dynamic function module after model initialization.

        Parameters
        ----------
        __context : Any
            Pydantic post-init context.

        Returns
        -------
        None
            Populates the internal loaded module reference.
        """
        function_path_values: Any = self.io_man.args.get("function_path")
        if (
            not isinstance(function_path_values, list)
            or not function_path_values
            or not isinstance(function_path_values[0], str)
        ):
            raise ValueError("Missing or invalid function_path argument")
        function_path: str = function_path_values[0]
        spec = importlib.util.spec_from_file_location(
                "functions", function_path
            )
        if spec is None:
            raise ImportError(
                f"Unable to load module spec from {function_path}"
            )
        if (
            spec.loader is None
            or not isinstance(spec.loader, importlib.abc.Loader)
        ):
            raise ImportError(
                f"Missing loader for module spec: {function_path}"
            )
        self._functions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._functions)

    @property
    def functions(self) -> types.ModuleType:
        """Return the loaded module containing callable functions.

        Returns
        -------
        types.ModuleType
            Imported module object.
        """
        return self._functions

    def execute_function(
                self,
                function_name: str,
                params: dict[str, Any]
            ) -> dict[str, Any] | None:
        """Execute a named function and capture stdout and return value.

        Parameters
        ----------
        function_name : str
            Attribute name of the function to call.
        params : dict[str, Any]
            Keyword arguments passed to the function.

        Returns
        -------
        dict[str, Any] | None
            Result payload with output and return value, or ``None`` when the
            function cannot be resolved.
        """
        output: io.StringIO = io.StringIO()
        try:
            target: Any = getattr(self.functions, function_name)
            if not callable(target):
                return None
            function: Callable[..., Any] = cast(Callable[..., Any], target)
        except Exception:
            return None
        with contextlib.redirect_stdout(output):
            function_return: Any = None
            function_error: Exception | None = None
            try:
                function_return = function(**params)
            except Exception as e:
                function_error = e
            return_dict: dict[str, Any] = {
                    "function_name": function_name,
                    "params": params,
                    "return": function_return,
                    "output": output.getvalue()
                }
            if function_error:
                return_dict.update({"error": function_error})
            return return_dict
