from unittest.mock import MagicMock, patch

import pytest

from fm.embeddings import EmbeddingResult, embed_text, get_available_provider


class TestEmbeddingResult:
    def test_create(self) -> None:
        result = EmbeddingResult(vector=[0.1, 0.2, 0.3], provider="voyage")
        assert result.provider == "voyage"
        assert len(result.vector) == 3


class TestGetAvailableProvider:
    @patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"})
    def test_voyage_available(self) -> None:
        provider = get_available_provider()
        assert provider in ("voyage", "huggingface")

    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_when_no_keys(self) -> None:
        provider = get_available_provider()
        assert provider in ("huggingface", None)


class TestEmbedText:
    def test_returns_embedding_result(self) -> None:
        with patch("fm.embeddings._embed_voyage") as mock_voyage:
            mock_voyage.return_value = [0.1, 0.2, 0.3]
            with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
                result = embed_text("test query", provider="voyage")
                assert result is not None
                assert result.provider == "voyage"
                assert len(result.vector) == 3

    def test_returns_none_when_provider_unavailable(self) -> None:
        result = embed_text("test query", provider=None)
        assert result is None
