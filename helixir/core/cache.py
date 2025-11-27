"""
Thread-safe LRU Cache implementation for Helixir.

This module provides a proper LRU (Least Recently Used) cache using OrderedDict
that correctly moves items to the end on access (cache hit).

Features:
- Thread-safe operations with RLock
- Proper LRU eviction (removes least recently used)
- Generic typing support
- O(1) get/set/delete operations
- Optional TTL support
"""

from collections import OrderedDict
from threading import RLock
import time
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

K = TypeVar("K")
V = TypeVar("V")


class LRUCache[K, V]:
    """
    Thread-safe LRU Cache with proper access-order maintenance.

    Unlike naive dict-based implementations, this correctly moves items
    to the end when accessed, ensuring true LRU behavior.

    Example:
        >>> cache = LRUCache[str, int](maxsize=2)
        >>> cache.set("a", 1)
        >>> cache.set("b", 2)
        >>> cache.get("a")
        1
        >>> cache.set("c", 3)
        >>> cache.get("b")
        None
    """

    def __init__(
        self,
        maxsize: int = 1000,
        ttl: float | None = None,
        on_evict: Callable[[K, V], None] | None = None,
    ):
        """
        Initialize LRU Cache.

        Args:
            maxsize: Maximum number of items in cache
            ttl: Optional time-to-live in seconds
            on_evict: Optional callback when item is evicted
        """
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")

        self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._on_evict = on_evict
        self._lock = RLock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K, default: V | None = None) -> V | None:
        """
        Get value from cache, moving it to end (most recently used).

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default

            value, timestamp = self._cache[key]

            if self._ttl is not None:
                age = time.time() - timestamp
                if age > self._ttl:
                    del self._cache[key]
                    self._misses += 1
                    return default

            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        """
        Set value in cache.

        If key exists, updates value and moves to end.
        If cache is full, evicts least recently used item.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            timestamp = time.time()

            if key in self._cache:
                self._cache[key] = (value, timestamp)
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._evict_oldest()

                self._cache[key] = (value, timestamp)

    def delete(self, key: K) -> bool:
        """
        Delete item from cache.

        Args:
            key: Cache key

        Returns:
            True if item was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                value, _ = self._cache.pop(key)
                if self._on_evict:
                    self._on_evict(key, value)
                return True
            return False

    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            if self._on_evict:
                for key, (value, _) in self._cache.items():
                    self._on_evict(key, value)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def _evict_oldest(self) -> None:
        """Evict least recently used item (FIFO from OrderedDict)."""
        if not self._cache:
            return

        key, (value, _) = self._cache.popitem(last=False)
        self._evictions += 1

        if self._on_evict:
            self._on_evict(key, value)

    def __len__(self) -> int:
        """Return number of items in cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache (doesn't update access order)."""
        with self._lock:
            return key in self._cache

    def values(self) -> list[V]:
        """
        Get all cached values (doesn't update access order).

        Returns:
            List of all cached values
        """
        with self._lock:
            return [value for value, _ in self._cache.values()]

    def keys(self) -> list[K]:
        """
        Get all cached keys.

        Returns:
            List of all cached keys
        """
        with self._lock:
            return list(self._cache.keys())

    def items(self) -> list[tuple[K, V]]:
        """
        Get all cached key-value pairs (doesn't update access order).

        Returns:
            List of (key, value) tuples
        """
        with self._lock:
            return [(key, value) for key, (value, _) in self._cache.items()]

    @property
    def maxsize(self) -> int:
        """Maximum cache size."""
        return self._maxsize

    @property
    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, evictions, size, hit_rate
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": f"{hit_rate:.2f}%",
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0


class EmbeddingCache(LRUCache[str, list[float]]):
    """
    Specialized LRU cache for embedding vectors.

    Uses text as key and embedding vector as value.
    Provides semantic-aware caching for LLM embeddings.

    Example:
        >>> cache = EmbeddingCache(maxsize=1000, ttl=3600)
        >>> cache.set("Paris is the capital", [0.1, 0.2, ...])
        >>> vec = cache.get("Paris is the capital")
    """

    def __init__(self, maxsize: int = 1000, ttl: float | None = 3600):
        """
        Initialize embedding cache.

        Args:
            maxsize: Maximum number of embeddings to cache
            ttl: Time-to-live in seconds (default 1 hour)
        """
        super().__init__(maxsize=maxsize, ttl=ttl)

    def get_vector_dim(self) -> int | None:
        """Get dimensionality of cached vectors."""
        with self._lock:
            if not self._cache:
                return None
            first_value, _ = next(iter(self._cache.values()))
            return len(first_value)
