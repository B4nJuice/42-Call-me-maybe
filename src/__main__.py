import asyncio
from .app.prompt_app import PromptApplication


async def run() -> None:
    """Run the prompt application entrypoint.

    Returns
    -------
    None
        This coroutine runs the application and returns when complete.
    """
    application: PromptApplication = PromptApplication()
    await application.run()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(e)
