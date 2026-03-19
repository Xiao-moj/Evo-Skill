"""
Local BGE-M3 embedding model.

Loads BAAI/bge-m3 from the local HuggingFace cache
(~/.cache/huggingface/hub/models--BAAI--bge-m3).

Dependencies (install one of):
  pip install FlagEmbedding          # preferred, official BAAI library
  pip install sentence-transformers  # fallback

BGE-M3 outputs 1024-dim normalized vectors.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

from .base import EmbeddingModel


@dataclass
class BGEM3Embedding(EmbeddingModel):
    """
    Local BGE-M3 embedding model via FlagEmbedding or sentence-transformers.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
                            Defaults to "BAAI/bge-m3" (uses HF cache automatically).
        batch_size:         Inference batch size. Lower if OOM.
        max_length:         Max token length (BGE-M3 supports up to 8192).
        use_fp16:           Use float16 for faster inference on GPU.
        device:             "cuda", "cpu", or None (auto-detect).
    """

    model_name_or_path: str = "BAAI/bge-m3"
    batch_size: int = 12
    max_length: int = 512
    use_fp16: bool = False
    device: Optional[str] = None

    # Internal state — not part of the public API
    _model: object = field(default=None, init=False, repr=False)
    _backend: str = field(default="", init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _load(self) -> None:
        """Lazy-load the model on first use (thread-safe)."""
        with self._lock:
            if self._model is not None:
                return

            # Try FlagEmbedding first (official BAAI library, more features)
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore
                kwargs = {
                    "use_fp16": self.use_fp16,
                }
                if self.device is not None:
                    kwargs["device"] = self.device
                self._model = BGEM3FlagModel(self.model_name_or_path, **kwargs)
                self._backend = "flagembedding"
                return
            except ImportError:
                pass

            # Fallback: sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                import torch

                device = self.device
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(
                    self.model_name_or_path,
                    device=device,
                )
                self._backend = "sentence_transformers"
                return
            except ImportError:
                pass

            raise RuntimeError(
                "BGE-M3 requires either FlagEmbedding or sentence-transformers.\n"
                "  pip install FlagEmbedding\n"
                "  # or\n"
                "  pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns a list of 1024-dim float vectors."""
        if not texts:
            return []

        self._load()

        cleaned = [str(t or "").strip() for t in texts]

        if self._backend == "flagembedding":
            # BGEM3FlagModel.encode returns a dict with key 'dense_vecs'
            result = self._model.encode(  # type: ignore[union-attr]
                cleaned,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = result["dense_vecs"]
            return [[float(x) for x in v] for v in vecs]

        if self._backend == "sentence_transformers":
            import numpy as np
            vecs = self._model.encode(  # type: ignore[union-attr]
                cleaned,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in v] for v in vecs]

        raise RuntimeError("Model not loaded")
