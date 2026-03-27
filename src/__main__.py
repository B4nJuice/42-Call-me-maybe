import asyncio
from .app.prompt_app import PromptApplication


class ApplicationEntrypoint:
    def __init__(self) -> None:
        self.application: PromptApplication = PromptApplication()

    async def run(self) -> None:
        await self.application.run()


if __name__ == "__main__":
    asyncio.run(ApplicationEntrypoint().run())
