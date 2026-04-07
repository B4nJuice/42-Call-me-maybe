from enum import Enum
from pydantic import BaseModel


class Colors(Enum):
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[31m"


class TerminalStyler(BaseModel):
    @staticmethod
    def clear_current_line() -> None:
        print("\x1b[2K\x1b[G", end="", flush=True)

    @staticmethod
    def colored_text(colors: list[Colors], text: str) -> str:
        rendered_text: str = "".join([color.value for color in colors])
        rendered_text += text
        rendered_text += Colors.RESET.value
        return rendered_text
