"""Vector indexing backends — Evo-skill edition."""

from .base import VectorIndex, VectorStore
from .factory import build_vector_index, list_vector_backends, register_vector_backend
from .flat import FlatFileVectorIndex

__all__ = [
    "VectorIndex",
    "VectorStore",
    "FlatFileVectorIndex",
    "build_vector_index",
    "register_vector_backend",
    "list_vector_backends",
]
