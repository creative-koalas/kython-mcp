import asyncio

from .server import server, session_store


async def _shutdown():
    await session_store.close_all()


def main() -> None:
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(_shutdown())
