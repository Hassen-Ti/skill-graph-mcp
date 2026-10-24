# core/embeddings.py
import logging
import os
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 3072
# 8191 token limit for text-embedding-3-large
_MAX_TOKENS = 8191
INDEX_NAME = "skill_description_embedding"

_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def _truncate_to_token_limit(text: str, max_tokens: int = _MAX_TOKENS, label: str | None = None) -> str:
    """Truncate text to stay within the model's token limit using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("text-embedding-3-large")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        logger.warning(
            "Truncating embedding input for %s: %d tokens -> %d (exceeds %s's %d-token limit). "
            "The embedding will not represent the full content.",
            label or "<unlabeled>", len(tokens), max_tokens, EMBEDDING_MODEL, max_tokens,
        )
        return enc.decode(tokens[:max_tokens])
    except Exception:
        # Fallback: rough char-based truncation (~4 chars/token)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            logger.warning(
                "Truncating embedding input for %s via char-based fallback (tiktoken unavailable): "
                "%d chars -> %d.",
                label or "<unlabeled>", len(text), max_chars,
            )
        return text[:max_chars]


async def embed_text(text: str, label: str | None = None) -> list[float]:
    client = _get_openai_client()
    text = _truncate_to_token_limit(text, label=label)
    response = await client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding
