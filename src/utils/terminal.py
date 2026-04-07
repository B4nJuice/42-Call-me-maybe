from enum import Enum
from pydantic import BaseModel


class Colors(Enum):
    """ANSI color/style escape sequences used for terminal rendering."""

    BOLD = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[31m"


class TerminalStyler(BaseModel):
    """Provide terminal styling helpers for line control and colors."""

    @staticmethod
    def clear_current_line() -> None:
        """Clear the current terminal line and move cursor to line start.

        Returns
        -------
        None
            Writes ANSI control sequences to stdout.
        """
        print("\x1b[2K\x1b[G", end="", flush=True)

    @staticmethod
    def colored_text(colors: list[Colors], text: str) -> str:
        """Wrap text with ANSI color/style sequences.

        Parameters
        ----------
        colors : list[Colors]
            Ordered list of styles to apply.
        text : str
            Text to format.

        Returns
        -------
        str
            Styled text including a trailing reset sequence.
        """
        rendered_text: str = "".join([color.value for color in colors])
        rendered_text += text
        rendered_text += Colors.RESET.value
        return rendered_text
