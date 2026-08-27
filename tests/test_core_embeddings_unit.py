# tests/test_core_embeddings_unit.py
"""
Unit tests for core/embeddings.py. All OpenAI calls are mocked — no network required.
"""
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock


def _fake_embedding(dims: int = 3072) -> list[float]:
    return [0.1] * dims


def _make_openai_response(embedding: list[float]):
    embedding_obj = MagicMock()
    embedding_obj.embedding = embedding
    response = MagicMock()
    response.data = [embedding_obj]
    return response


@pytest.mark.asyncio
async def test_embed_text_returns_3072_floats(mocker):
    fake_vec = _fake_embedding(3072)
    mock_create = AsyncMock(return_value=_make_openai_response(fake_vec))
    mocker.patch(
        "core.embeddings._get_openai_client",
        return_value=MagicMock(embeddings=MagicMock(create=mock_create)),
    )

    from core.embeddings import embed_text

    result = await embed_text("hello world")

    assert isinstance(result, list)
    assert len(result) == 3072


@pytest.mark.asyncio
async def test_embed_text_calls_correct_model(mocker):
    fake_vec = _fake_embedding()
    mock_create = AsyncMock(return_value=_make_openai_response(fake_vec))
    mocker.patch(
        "core.embeddings._get_openai_client",
        return_value=MagicMock(embeddings=MagicMock(create=mock_create)),
    )

    from core.embeddings import embed_text, EMBEDDING_MODEL

    await embed_text("some query text")

    mock_create.assert_awaited_once()
    assert mock_create.call_args.kwargs.get("model") == EMBEDDING_MODEL == "text-embedding-3-large"


def test_truncate_no_op_under_limit():
    from core.embeddings import _truncate_to_token_limit

    text = "short text"
    assert _truncate_to_token_limit(text, max_tokens=8191) == text


def test_truncate_shortens_oversized_text():
    from core.embeddings import _truncate_to_token_limit

    text = "word " * 20000  # far more than any reasonable token limit
    result = _truncate_to_token_limit(text, max_tokens=100)

    assert result != text
    assert len(result) < len(text)


def test_truncate_logs_warning_when_it_fires(caplog):
    """The bug this session found: truncation used to be silent. Must now warn, with the label."""
    from core.embeddings import _truncate_to_token_limit

    with caplog.at_level(logging.WARNING, logger="core.embeddings"):
        _truncate_to_token_limit("word " * 20000, max_tokens=100, label="game-development")

    assert any("game-development" in r.message for r in caplog.records)
    assert any("Truncating" in r.message for r in caplog.records)


def test_truncate_does_not_log_when_under_limit(caplog):
    from core.embeddings import _truncate_to_token_limit

    with caplog.at_level(logging.WARNING, logger="core.embeddings"):
        _truncate_to_token_limit("short text", max_tokens=8191, label="tiny-skill")

    assert not any("tiny-skill" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_embed_text_passes_label_through_to_truncation(mocker):
    """embed_text must forward its label so a truncation warning can identify the skill."""
    fake_vec = _fake_embedding()
    mock_create = AsyncMock(return_value=_make_openai_response(fake_vec))
    mocker.patch(
        "core.embeddings._get_openai_client",
        return_value=MagicMock(embeddings=MagicMock(create=mock_create)),
    )
    mock_truncate = mocker.patch(
        "core.embeddings._truncate_to_token_limit", return_value="short",
    )

    from core.embeddings import embed_text

    await embed_text("some text", label="my-skill")

    assert mock_truncate.call_args.kwargs.get("label") == "my-skill"
