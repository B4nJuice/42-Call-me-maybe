from ..ui.terminal import Colors, TerminalStyler


class LogConsole:
    @staticmethod
    def clear_current_line() -> None:
        TerminalStyler.clear_current_line()

    @staticmethod
    def colored_text(colors: list[Colors], text: str) -> str:
        return TerminalStyler.colored_text(colors, text)
