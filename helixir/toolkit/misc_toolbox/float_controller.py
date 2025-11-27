"""
Float Controller - Event-driven system for tracking code execution.

Floats are execution markers that help:
1. Track the entire execution flow
2. Debug issues in E2E tests
3. Collect performance metrics
4. Verify that code executed

Pattern: Event-Driven Development
- Each float = event
- Controller collects events
- Can be enabled/disabled via parameter
- In tests: enabled=True → check all events
- In production: enabled=False → zero overhead

Usage:
    >>> from helixir.toolkit.misc_toolbox.float_controller import FloatController
    >>>
    >>> fc = FloatController(enabled=True)
    >>> fc.float("extraction_start", user_id="user123")
    >>> fc.float("extraction_success", count=5)
    >>>
    >>> assert fc.has_float("extraction_start")
    >>> assert fc.get_float("extraction_success")["count"] == 5
    >>>
    >>> fc = FloatController(enabled=False)
    >>> fc.float("anything", data="ignored")
"""

from collections import defaultdict
from datetime import datetime
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class FloatEvent:
    """
    Single float event.

    Attributes:
        float_id: Unique float identifier
        name: Float name (e.g., "extraction_start")
        timestamp: When float occurred
        data: Additional data attached to float
        trace_id: Optional trace ID for correlation
    """

    def __init__(self, name: str, data: dict[str, Any] | None = None, trace_id: str | None = None):
        self.float_id = str(uuid4())
        self.name = name
        self.timestamp = datetime.now()
        self.data = data or {}
        self.trace_id = trace_id

    def __repr__(self) -> str:
        return (
            f"FloatEvent(name={self.name!r}, "
            f"timestamp={self.timestamp.isoformat()}, "
            f"data={self.data})"
        )


class FloatController:
    """
    Event-driven controller for float markers.

    Controls float system globally:
    - Enable/disable floats (for production vs tests)
    - Collect float events
    - Query floats by name/pattern
    - Generate reports

    Singleton pattern: one controller per application

    Usage:
        >>>
        >>> fc = FloatController.get_instance(enabled=True)
        >>>
        >>> fc.float("user.login", user_id="123")
        >>> fc.float("memory.extract", count=5)
        >>>
        >>> events = fc.get_floats("memory.*")
        >>> assert len(events) == 1
        >>>
        >>> fc.clear()
    """

    _instance: FloatController | None = None

    def __init__(self, enabled: bool = False):
        """
        Initialize float controller.

        Args:
            enabled: Whether to collect floats (True in tests, False in prod)
        """
        self.enabled = enabled
        self._floats: list[FloatEvent] = []
        self._floats_by_name: dict[str, list[FloatEvent]] = defaultdict(list)
        self._current_trace_id: str | None = None

        logger.info(f"FloatController initialized: enabled={enabled}")

    @classmethod
    def get_instance(cls, enabled: bool | None = None) -> FloatController:
        """
        Get singleton instance.

        Args:
            enabled: Override enabled state (only on first call)

        Returns:
            Global FloatController instance
        """
        if cls._instance is None:
            cls._instance = cls(enabled=enabled if enabled is not None else False)
        elif enabled is not None:
            cls._instance.enabled = enabled

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for tests)."""
        cls._instance = None

    def enable(self) -> None:
        """Enable float collection."""
        self.enabled = True
        logger.info("FloatController: ENABLED")

    def disable(self) -> None:
        """Disable float collection (for production)."""
        self.enabled = False
        logger.info("FloatController: DISABLED")

    def set_trace_id(self, trace_id: str) -> None:
        """Set current trace ID for correlation."""
        self._current_trace_id = trace_id

    def clear_trace_id(self) -> None:
        """Clear current trace ID."""
        self._current_trace_id = None

    def float(self, event_name: str, **data: Any) -> FloatEvent | None:
        """
        Emit a float event.

        Args:
            event_name: Float event name (e.g., "extraction.start")
            **data: Additional data to attach (can include 'name' key)

        Returns:
            FloatEvent if enabled, None otherwise

        Example:
            >>> fc.float("memory.add", user_id="user123", count=5)
            >>> fc.float("decision.made", operation="ADD", confidence=95)
            >>> fc.float("entity.create", name="John", entity_type="Person")
        """
        if not self.enabled:
            return None

        event = FloatEvent(name=event_name, data=data, trace_id=self._current_trace_id)

        self._floats.append(event)
        self._floats_by_name[event_name].append(event)

        logger.debug(
            f"🎈 FLOAT[{event_name}] {data if data else ''} (trace={self._current_trace_id})"
        )

        return event

    def has_float(self, name: str) -> bool:
        """
        Check if float occurred.

        Args:
            name: Float name

        Returns:
            True if float was emitted
        """
        return name in self._floats_by_name

    def get_floats(self, pattern: str | None = None) -> list[FloatEvent]:
        """
        Get all floats, optionally filtered by pattern.

        Args:
            pattern: Optional pattern (e.g., "memory.*" matches "memory.add", "memory.search")

        Returns:
            List of matching FloatEvents

        Example:
            >>> events = fc.get_floats("memory.*")
            >>> events = fc.get_floats()
        """
        if pattern is None:
            return self._floats.copy()

        if "*" in pattern:
            prefix = pattern.replace("*", "")
            return [event for event in self._floats if event.name.startswith(prefix)]
        return self._floats_by_name.get(pattern, []).copy()

    def get_float(self, name: str, index: int = 0) -> FloatEvent | None:
        """
        Get specific float by name and index.

        Args:
            name: Float name
            index: Index if multiple floats with same name (default: 0)

        Returns:
            FloatEvent or None
        """
        events = self._floats_by_name.get(name, [])
        if index < len(events):
            return events[index]
        return None

    def count_floats(self, pattern: str | None = None) -> int:
        """
        Count floats matching pattern.

        Args:
            pattern: Optional pattern

        Returns:
            Count of matching floats
        """
        return len(self.get_floats(pattern))

    def clear(self) -> None:
        """Clear all collected floats."""
        self._floats.clear()
        self._floats_by_name.clear()
        self._current_trace_id = None
        logger.debug("FloatController: cleared")

    def get_report(self) -> dict[str, Any]:
        """
        Get float report.

        Returns:
            Dict with statistics about collected floats
        """
        float_counts = {name: len(events) for name, events in self._floats_by_name.items()}

        return {
            "enabled": self.enabled,
            "total_floats": len(self._floats),
            "unique_names": len(self._floats_by_name),
            "float_counts": float_counts,
            "current_trace_id": self._current_trace_id,
        }

    def print_report(self) -> None:
        """Print float report to console."""
        report = self.get_report()

        for _name, _count in sorted(report["float_counts"].items()):
            pass

    def __repr__(self) -> str:
        return f"FloatController(enabled={self.enabled}, floats={len(self._floats)})"


def get_float_controller(enabled: bool | None = None) -> FloatController:
    """
    Get global float controller instance.

    Args:
        enabled: Override enabled state

    Returns:
        FloatController singleton

    Example:
        >>> fc = get_float_controller(enabled=True)
        >>> fc.float("test.start")
    """
    return FloatController.get_instance(enabled=enabled)


def float_event(event_name: str, **data: Any) -> FloatEvent | None:
    """
    Quick emit a float event using global controller.

    Args:
        event_name: Float event name (e.g., "user.login", "memory.add")
        **data: Additional data (can include 'name' key without conflict)

    Returns:
        FloatEvent if enabled, None otherwise

    Example:
        >>> from helixir.toolkit.misc_toolbox.float_controller import float_event
        >>> float_event("user.login", user_id="123")
        >>> float_event("entity.create", name="John", entity_type="Person")
    """
    return FloatController.get_instance().float(event_name, **data)


class FloatContext:
    """
    Context manager for float collection in tests.

    Usage:
        >>> with FloatContext() as fc:
        ...     client.add_memory(...)
        ...
        ...     assert fc.has_float("extraction.start")
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.fc = FloatController.get_instance()
        self._old_enabled = self.fc.enabled

    def __enter__(self) -> FloatController:
        self.fc.enabled = self.enabled
        self.fc.clear()
        return self.fc

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fc.enabled = self._old_enabled
        return False
