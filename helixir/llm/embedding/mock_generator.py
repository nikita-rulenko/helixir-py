"""Mock Embedding Generator for testing."""

from typing import Any

from helixir.llm.embedding.base import BaseEmbeddingGenerator
import numpy as np


class MockEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Mock embedding generator for testing.

    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.call_count = 0

    def generate(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate mock embedding.

        Args:
            text: Input text

        Returns:
            Deterministic embedding vector
        """
        self.call_count += 1

        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)

        vector = rng.standard_normal(self.dimensions)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def generate_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        return [self.generate(text, **kwargs) for text in texts]

    def reset(self):
        """Reset call counter."""
        self.call_count = 0
