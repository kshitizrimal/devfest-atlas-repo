"""READ AND WRITE: Gemini 3.8 Flash, asked for typed JSON.

Every call hands Gemini a Pydantic schema, so what comes back is a typed object, never free text to parse.
Thinking stays low: reading a page is copying, and writing from evidence needs little reasoning.
"""
from google import genai
from google.genai import types
from pydantic import BaseModel

from .config import GEMINI_API_KEY, GEMINI_MODEL

_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def typed(parts: list, schema: type[BaseModel], thinking: str = "LOW") -> BaseModel:
    """parts: text and images (types.Part). Returns an instance of schema."""
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,                                   # the shape it must return
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
    )
    r = await client().aio.models.generate_content(model=GEMINI_MODEL, contents=parts, config=cfg)
    return r.parsed if isinstance(r.parsed, schema) else schema.model_validate_json(r.text)


def image(data: bytes, mime: str = "image/png") -> types.Part:
    return types.Part.from_bytes(data=data, mime_type=mime)
