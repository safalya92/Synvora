from __future__ import annotations

from unittest.mock import patch

from config import MODEL_DOWNLOAD_CONFIG
from indexing.embedding_manager import EmbeddingManager


def test_embedding_manager_uses_local_cache_when_offline(tmp_path):
    original = MODEL_DOWNLOAD_CONFIG.copy()
    MODEL_DOWNLOAD_CONFIG["local_files_only"] = True
    MODEL_DOWNLOAD_CONFIG["cache_dir"] = str(tmp_path)

    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls["text_model"] = {"model_name": model_name, "kwargs": kwargs}

        def encode(self, *args, **kwargs):
            return [0.0]

        def to(self, device):
            return self

    class FakeCLIPModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["clip_model"] = {"model_name": model_name, "kwargs": kwargs}
            return cls()

        def to(self, device):
            return self

    class FakeCLIPProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["clip_processor"] = {"model_name": model_name, "kwargs": kwargs}
            return cls()

    try:
        with patch("indexing.embedding_manager.SentenceTransformer", FakeSentenceTransformer), patch(
            "indexing.embedding_manager.CLIPModel", FakeCLIPModel
        ), patch("indexing.embedding_manager.CLIPProcessor", FakeCLIPProcessor):
            EmbeddingManager(device="cpu", cache_dir=tmp_path)

        assert calls["text_model"]["kwargs"]["local_files_only"] is True
        assert calls["text_model"]["kwargs"]["cache_folder"] == str(tmp_path)
        assert calls["clip_model"]["kwargs"]["local_files_only"] is True
        assert calls["clip_model"]["kwargs"]["cache_dir"] == str(tmp_path)
        assert calls["clip_processor"]["kwargs"]["local_files_only"] is True
        assert calls["clip_processor"]["kwargs"]["cache_dir"] == str(tmp_path)
    finally:
        MODEL_DOWNLOAD_CONFIG.clear()
        MODEL_DOWNLOAD_CONFIG.update(original)
