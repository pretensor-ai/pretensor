"""Local ONNX embedding clients for cross-database entity resolution (optional ``[embeddings]`` extra)."""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Callable, Protocol, cast, runtime_checkable

__all__ = [
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_MODEL_REVISION",
    "EMBEDDING_DIM",
    "EmbeddingClient",
    "LocalEmbeddingClient",
    "NullEmbeddingClient",
    "cosine_similarity",
    "embeddings_disabled_via_env",
    "embeddings_extra_installed",
    "resolve_embeddings_auto",
    "format_entity_text",
    "get_default_embedding_client",
    "verify_embeddings_extra_installed",
]

EMBEDDING_MODEL_ID = "Snowflake/snowflake-arctic-embed-xs"
# Pinned to a specific HuggingFace commit so entity-resolution scores stay reproducible
# across runs even if the model is updated upstream. Bump explicitly when validating a
# new revision against the cross-DB benchmark pairs.
EMBEDDING_MODEL_REVISION = "d8c86521100d3556476a063fc2342036d45c106f"
EMBEDDING_DIM = 384

_EMBEDDINGS_INSTALL_HINT = (
    "Install embedding dependencies with: pip install 'pretensor[embeddings]' "
    "(requires onnxruntime, huggingface_hub, numpy, transformers)."
)

_ENV_DISABLED_FLAG = "PRETENSOR_EMBEDDINGS_DISABLED"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def embeddings_disabled_via_env() -> bool:
    """Return True iff ``PRETENSOR_EMBEDDINGS_DISABLED`` is set to a truthy value.

    Honored by every embedding-consuming code path — index-time steps,
    intelligence toggles, and query-time MCP tools — so setting the variable
    forces the null path even when the ``[embeddings]`` extra is installed
    and toggles are on.  CI exercises an "extras installed but disabled"
    lane that must produce output byte-identical to the "extras absent"
    lane.  Truthy values are ``1``, ``true``, ``yes``, ``on``
    (case-insensitive).
    """
    return os.environ.get(_ENV_DISABLED_FLAG, "").strip().lower() in _TRUTHY


@runtime_checkable
class EmbeddingClient(Protocol):
    """Structural type for text embedding backends (sync ``embed``; usable with ``asyncio.to_thread``)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string (or an empty list when disabled)."""
        ...


class NullEmbeddingClient:
    """No-op client when embeddings are off (e.g. ``--embeddings`` not set). Callers treat ``[]`` as no similarity."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in ``[-1, 1]`` (1.0 for identical non-zero vectors, 0.0 when orthogonal or zero norm)."""
    if len(a) != len(b):
        msg = f"Vector length mismatch: {len(a)} vs {len(b)}"
        raise ValueError(msg)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def verify_embeddings_extra_installed() -> None:
    """Raise ``ImportError`` with an install hint when ``[embeddings]`` is missing.

    Public guard intended for callers (notably the CLI ``index`` and
    ``reindex`` commands) that want to surface a friendly install hint
    *before* spinning up a pipeline that would otherwise fail mid-run
    when ``LocalEmbeddingClient.embed()`` reaches its lazy import.
    Returns ``None`` on success.
    """
    _require_embeddings_imports()


def embeddings_extra_installed() -> bool:
    """Return True iff the optional ``[embeddings]`` dependencies import cleanly."""
    try:
        _require_embeddings_imports()
    except ImportError:
        return False
    return True


def resolve_embeddings_auto(requested: bool | None) -> bool:
    """Resolve the CLI's tri-state ``--embeddings/--no-embeddings`` flag.

    ``None`` (neither flag given) means AUTO: embeddings are on whenever
    the ``[embeddings]`` extra is installed and the
    ``PRETENSOR_EMBEDDINGS_DISABLED`` kill switch is not set.  Installing
    the extra is the opt-in; the base install never pays the cost.  An
    explicit ``True``/``False`` always wins — ``True`` is validated by the
    caller via :func:`verify_embeddings_extra_installed` so a missing
    extra errors with the install hint instead of silently downgrading.

    The AUTO behavior lives at the CLI boundary on purpose: library-level
    defaults (``EmbeddingsConfig()``) stay all-off so the null-path parity
    contract for direct ``GraphBuilder`` / ``run_intelligence_layer``
    callers is unchanged.
    """
    if requested is not None:
        return requested
    return embeddings_extra_installed() and not embeddings_disabled_via_env()


def _require_embeddings_imports() -> tuple[Any, Any, Callable[..., str], Any]:
    """Import optional deps; raise ImportError with install hint if missing."""
    try:
        np = importlib.import_module("numpy")
    except ImportError as e:
        raise ImportError(_EMBEDDINGS_INSTALL_HINT) from e
    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError as e:
        raise ImportError(_EMBEDDINGS_INSTALL_HINT) from e
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as e:
        raise ImportError(_EMBEDDINGS_INSTALL_HINT) from e
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as e:
        raise ImportError(_EMBEDDINGS_INSTALL_HINT) from e
    snapshot_download = cast(Callable[..., str], hub.snapshot_download)
    AutoTokenizer = transformers.AutoTokenizer
    return np, ort, snapshot_download, AutoTokenizer


class LocalEmbeddingClient:
    """Loads ``Snowflake/snowflake-arctic-embed-xs`` ONNX from the Hugging Face cache (384-dim CLS embeddings).

    The HuggingFace revision is pinned via ``EMBEDDING_MODEL_REVISION`` so that
    entity-resolution scores are reproducible across runs. Override with the
    ``model_revision`` kwarg only for tests or deliberate model bumps.
    """

    def __init__(
        self,
        *,
        model_id: str = EMBEDDING_MODEL_ID,
        model_revision: str = EMBEDDING_MODEL_REVISION,
        onnx_filename: str = "onnx/model.onnx",
        _session_factory: Any | None = None,
        _model_dir: Path | None = None,
    ) -> None:
        self._model_id = model_id
        self._revision = model_revision
        self._onnx_relative = onnx_filename
        self._session_factory = _session_factory
        self._model_dir_override = _model_dir
        self._np: Any | None = None
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._input_names: tuple[str, ...] | None = None
        self._output_index_cls: int = 0

    def _ensure_loaded(self) -> None:
        if (
            self._session is not None
            and self._tokenizer is not None
            and self._np is not None
        ):
            return
        np, ort, snapshot_download, AutoTokenizer = _require_embeddings_imports()
        self._np = np
        model_dir = self._model_dir_override
        if model_dir is None:
            cache_dir = snapshot_download(
                repo_id=self._model_id,
                revision=self._revision,
                local_files_only=False,
                # Only the ONNX graph + tokenizer/config files are read;
                # without the filter the full repo snapshot (PyTorch
                # weights, every quantized ONNX variant) is ~4x larger on
                # the cold first download.
                allow_patterns=[
                    self._onnx_relative,
                    "*.json",
                    "*.txt",
                    "1_Pooling/*",
                ],
            )
            model_dir = Path(cache_dir)
        onnx_path = model_dir / self._onnx_relative
        if not onnx_path.is_file():
            msg = f"ONNX model not found at {onnx_path}"
            raise FileNotFoundError(msg)

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        factory = self._session_factory or ort.InferenceSession
        self._session = factory(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        sess = self._session
        assert sess is not None
        inputs = sess.get_inputs()
        self._input_names = tuple(inp.name for inp in inputs)
        outputs = sess.get_outputs()
        out_names = [o.name for o in outputs]
        if "last_hidden_state" in out_names:
            self._output_index_cls = out_names.index("last_hidden_state")
        else:
            self._output_index_cls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        np = self._np
        assert (
            np is not None and self._tokenizer is not None and self._session is not None
        )
        tok = self._tokenizer
        encoded = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feed: dict[str, Any] = {}
        for name in self._input_names or ():
            if name == "input_ids":
                feed[name] = encoded["input_ids"].astype(np.int64)
            elif name == "attention_mask":
                feed[name] = encoded["attention_mask"].astype(np.int64)
            elif name == "token_type_ids" and "token_type_ids" in encoded:
                feed[name] = encoded["token_type_ids"].astype(np.int64)
        if not feed:
            msg = "Could not map tokenizer outputs to ONNX inputs"
            raise RuntimeError(msg)

        outputs = self._session.run(None, feed)
        last_hidden = np.asarray(outputs[self._output_index_cls], dtype=np.float32)
        if last_hidden.ndim == 3:
            # CLS token at index 0 (per Snowflake / BERT-style pooling).
            pooled = last_hidden[:, 0, :]
        elif last_hidden.ndim == 2:
            # Already-pooled export (e.g. a ``sentence_embedding`` output):
            # shape [batch, dim], nothing to slice.
            pooled = last_hidden
        else:
            msg = (
                "Unexpected ONNX output shape "
                f"{tuple(last_hidden.shape)}; expected [batch, seq, dim] "
                "or [batch, dim]"
            )
            raise RuntimeError(msg)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        pooled = pooled / norms
        return [row.astype(float).tolist() for row in pooled]


def format_entity_text(entity_name: str, column_names: list[str]) -> str:
    """Format schema text for embedding: ``\"{entity}: col1, col2\"`` (short, within model length)."""
    cols = ", ".join(column_names)
    return f"{entity_name}: {cols}"


# Process-wide cached default ``LocalEmbeddingClient``.  Every consumer
# (compile_metric, query rerank, traverse tie-break, semantic_search,
# role-vote, EmbeddingIndexStep) goes through ``get_default_embedding_client``
# so the multi-MB ONNX session/tokenizer state loads at most once per
# process.  Constructing ``LocalEmbeddingClient()`` is cheap (no I/O until
# ``embed()`` is called), but the model download + ``InferenceSession``
# init on first ``embed()`` is dominant on cold runs; sharing one instance
# keeps that cost amortized across all embedding-driven tools.
_DEFAULT_CLIENT: LocalEmbeddingClient | None = None


def get_default_embedding_client() -> LocalEmbeddingClient:
    """Return the process-wide cached ``LocalEmbeddingClient``.

    Constructs one on first call; every subsequent call returns the
    same instance.  Tests that need to reset the cache (e.g. to swap a
    stub embedder per test) should call
    :func:`_reset_default_embedding_client_for_tests` in an autouse
    fixture.
    """
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LocalEmbeddingClient()
    return _DEFAULT_CLIENT


def _reset_default_embedding_client_for_tests() -> None:
    """Test helper: drop the module-level default-client cache."""
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = None
