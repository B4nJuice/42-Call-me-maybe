from .terminal import Colors, TerminalStyler


class PromptTableRenderer:
    def __init__(self, prompt_texts: list[str]) -> None:
        self.prompt_texts: list[str] = prompt_texts
        self.statuses: list[str] = ["pending"] * len(prompt_texts)
        self.tokens: list[int] = [0] * len(prompt_texts)
        self.rendered_lines: int = 0

    def set_status(self, index: int, status: str) -> None:
        self.statuses[index] = status

    def set_token(self, index: int, token: int) -> None:
        self.tokens[index] = token

    def render(self, spin_char: str = "-") -> int:
        index_width: int = max(2, len(str(len(self.prompt_texts) - 1)))
        status_width: int = 10
        token_width: int = 6

        print(
            TerminalStyler.colored_text(
                [Colors.BOLD, Colors.CYAN], "Prompt queue"
            )
        )

        header_id: str = f"{'ID':<{index_width}}"
        header_status: str = f"{'STATUS':<{status_width}}"
        header_tokens: str = f"{'TOKENS':>{token_width}}"
        header_prompt: str = "PROMPT"

        header_plain: str = (
            f"{header_id} | {header_status} | {header_tokens} | {header_prompt}"
        )
        header: str = (
            f"{TerminalStyler.colored_text([Colors.BOLD], header_id)} | "
            f"{TerminalStyler.colored_text([Colors.BOLD], header_status)} | "
            f"{TerminalStyler.colored_text([Colors.BOLD], header_tokens)} | "
            f"{TerminalStyler.colored_text([Colors.BOLD], header_prompt)}"
        )
        separator: str = "-" * len(header_plain)

        print(header)
        print(TerminalStyler.colored_text([Colors.CYAN], separator))

        for idx, text in enumerate(self.prompt_texts):
            status_name: str = self.statuses[idx]
            status: str = self._format_status(status_name, spin_char)
            token_value: str = (
                "-" if self.statuses[idx] == "pending" else str(self.tokens[idx])
            )

            status_colors: dict[str, list[Colors]] = {
                "pending": [Colors.YELLOW],
                "running": [Colors.CYAN, Colors.BOLD],
                "done": [Colors.GREEN, Colors.BOLD],
                "error": [Colors.RED, Colors.BOLD],
            }

            colored_id: str = TerminalStyler.colored_text(
                [Colors.MAGENTA], f"{idx:<{index_width}}"
            )
            colored_status: str = TerminalStyler.colored_text(
                status_colors.get(status_name, [Colors.YELLOW]),
                f"{status:<{status_width}}",
            )
            colored_tokens: str = TerminalStyler.colored_text(
                [Colors.YELLOW], f"{token_value:>{token_width}}"
            )
            colored_prompt: str = TerminalStyler.colored_text(
                [Colors.CYAN], text
            )

            print(
                f"{colored_id} | "
                f"{colored_status} | "
                f"{colored_tokens} | "
                f"{colored_prompt}"
            )

        self.rendered_lines = len(self.prompt_texts) + 3
        return self.rendered_lines

    def redraw(self, spin_char: str = "-") -> int:
        print(f"\x1b[{self.rendered_lines}F", end="")
        for _ in range(self.rendered_lines):
            TerminalStyler.clear_current_line()
            print()
        print(f"\x1b[{self.rendered_lines}F", end="")
        return self.render(spin_char)

    @staticmethod
    def _format_status(status: str, spin_char: str) -> str:
        if status == "running":
            return f"running {spin_char}"
        return status
