"""Unit tests for ``pretensor.intelligence.embeddings`` (mocked ONNX; no Hub download)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pretensor.intelligence import embeddings as embeddings_mod
from pretensor.intelligence.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_REVISION,
    LocalEmbeddingClient,
    NullEmbeddingClient,
    cosine_similarity,
    format_entity_text,
)


def test_null_embedding_client_returns_empty() -> None:
    assert NullEmbeddingClient().embed(["anything"]) == []


def test_cosine_similarity_identical_is_one() -> None:
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_cosine_similarity_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_format_entity_text() -> None:
    assert format_entity_text("Customer", ["id", "name", "email"]) == (
        "Customer: id, name, email"
    )


def test_local_embedding_client_mocked_session_384_dim(tmp_path: Path) -> None:
    onnx_path = tmp_path / "onnx" / "model.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.touch()

    class _In:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Out:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            self.path = path

        def get_inputs(self) -> list[_In]:
            return [_In("input_ids"), _In("attention_mask")]

        def get_outputs(self) -> list[_Out]:
            return [_Out("last_hidden_state")]

        def run(self, _outputs: object, feed: dict[str, object]) -> list[np.ndarray]:
            input_ids = cast(np.ndarray, feed["input_ids"])
            batch = int(input_ids.shape[0])
            seq = int(input_ids.shape[1])
            hidden = EMBEDDING_DIM
            last = np.zeros((batch, seq, hidden), dtype=np.float32)
            last[:, 0, :] = 1.0
            return [last]

    class FakeTokenizer:
        def __call__(
            self,
            texts: list[str],
            *,
            padding: bool,
            truncation: bool,
            max_length: int,
            return_tensors: str,
        ) -> dict[str, np.ndarray]:
            batch = len(texts)
            return {
                "input_ids": np.ones((batch, 4), dtype=np.int64),
                "attention_mask": np.ones((batch, 4), dtype=np.int64),
            }

        @classmethod
        def from_pretrained(cls, _path: str) -> FakeTokenizer:
            return cls()

    client = LocalEmbeddingClient(
        _session_factory=FakeSession,
        _model_dir=tmp_path,
    )
    client._tokenizer = FakeTokenizer()
    client._np = np
    client._input_names = ("input_ids", "attention_mask")
    client._output_index_cls = 0
    client._session = FakeSession(str(onnx_path))

    out = client.embed(["Customer: id, name, email"])
    assert len(out) == 1
    assert len(out[0]) == EMBEDDING_DIM
    assert out[0][0] == pytest.approx(1.0 / (EMBEDDING_DIM**0.5))


def test_ensure_loaded_passes_pinned_revision_to_snapshot_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_ensure_loaded` must pin the HuggingFace revision for reproducibility."""
    onnx_path = tmp_path / "onnx" / "model.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.touch()

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _path: str) -> FakeTokenizer:
            return cls()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str) -> FakeTokenizer:
            return FakeTokenizer.from_pretrained(path)

    class _In:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Out:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            self.path = path

        def get_inputs(self) -> list[_In]:
            return [_In("input_ids"), _In("attention_mask")]

        def get_outputs(self) -> list[_Out]:
            return [_Out("last_hidden_state")]

    class FakeOrt:
        InferenceSession = FakeSession

    def fake_require_imports() -> tuple[object, object, object, object]:
        return np, FakeOrt, fake_snapshot_download, FakeAutoTokenizer

    monkeypatch.setattr(
        embeddings_mod, "_require_embeddings_imports", fake_require_imports
    )

    client = LocalEmbeddingClient()
    client._ensure_loaded()

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "Snowflake/snowflake-arctic-embed-xs"
    assert calls[0]["revision"] == EMBEDDING_MODEL_REVISION
    assert calls[0]["local_files_only"] is False


def test_local_embedding_client_accepts_revision_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit ``model_revision`` kwarg must override the pinned default."""
    onnx_path = tmp_path / "onnx" / "model.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.touch()

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _path: str) -> FakeTokenizer:
            return cls()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str) -> FakeTokenizer:
            return FakeTokenizer.from_pretrained(path)

    class _In:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Out:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            self.path = path

        def get_inputs(self) -> list[_In]:
            return [_In("input_ids")]

        def get_outputs(self) -> list[_Out]:
            return [_Out("last_hidden_state")]

    class FakeOrt:
        InferenceSession = FakeSession

    monkeypatch.setattr(
        embeddings_mod,
        "_require_embeddings_imports",
        lambda: (np, FakeOrt, fake_snapshot_download, FakeAutoTokenizer),
    )

    client = LocalEmbeddingClient(model_revision="deadbeef")
    client._ensure_loaded()
    assert calls[0]["revision"] == "deadbeef"


def test_onnx_missing_raises_before_path_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """ImportError references install hint (not missing ONNX path)."""
    real_import_module = importlib.import_module

    def guarded_import_module(name: str, package: str | None = None) -> object:
        if name == "onnxruntime":
            raise ImportError("missing onnx")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    client = LocalEmbeddingClient()
    with pytest.raises(ImportError) as exc:
        client.embed(["a"])
    assert "pretensor[embeddings]" in str(exc.value)
