import asyncio
from .app.prompt_app import PromptApplication


async def run() -> None:
    application: PromptApplication = PromptApplication()
    await application.run()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(e)
