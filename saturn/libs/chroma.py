import os
from typing import AsyncGenerator
from chromadb import AsyncHttpClient, AsyncClientAPI


async def get_chroma_connection() -> AsyncGenerator[AsyncClientAPI, None]:
    conn = await AsyncHttpClient(
        host=os.environ.get("CHROMA_HOST", "localhost"),
        port=os.environ.get("CHROMA_PORT", 8000),
    )

    yield conn
