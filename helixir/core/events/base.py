"""
Base Event System - Foundation for event-driven architecture.

Core Concepts:
- BaseEvent: Immutable event with timestamp and metadata
- EventHandler: Async callable that processes events
- EventListener: Decorator for registering handlers

Design Principles:
- Events are immutable (frozen dataclass)
- Handlers are async (Python 3.14 no-GIL!)
- Type-safe event dispatching
- Observable via Float integration
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID, uuid4

EventT = TypeVar("EventT", bound="BaseEvent")
HandlerFunc = Callable[[EventT], Coroutine[Any, Any, None]]


@dataclass(frozen=True, kw_only=True)
class BaseEvent(ABC):
    """
    Base class for all events in the system.

    Events are immutable records of something that happened.
    They contain all necessary context for handlers to process them.

    Attributes:
        event_id: Unique identifier for this event instance
        timestamp: When the event occurred
        correlation_id: Links related events across the pipeline
        metadata: Additional context (for Float tracking, debugging)
    """

    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def event_type(self) -> str:
        """
        Event type identifier (e.g., 'memory.created', 'chunk.linked').
        Used for handler registration and routing.
        """

    def to_float_data(self) -> dict[str, Any]:
        """
        Convert event to Float tracking data.
        Used for automatic observability integration.
        """
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            **self.metadata,
        }


class EventHandler(ABC):
    """
    Base class for event handlers.

    Handlers process events asynchronously.
    They can emit new events, call services, update state, etc.

    Handlers should be:
    - Idempotent (can be retried safely)
    - Fast (offload heavy work to background tasks)
    - Error-resilient (handle failures gracefully)
    """

    @abstractmethod
    async def handle(self, event: BaseEvent) -> None:
        """
        Process an event.

        Args:
            event: The event to process

        Raises:
            Exception: Handler errors are caught by EventBus for retry
        """

    @property
    @abstractmethod
    def event_types(self) -> list[str]:
        """
        List of event types this handler processes.
        Used by EventBus for routing.
        """

    @property
    def handler_name(self) -> str:
        """Human-readable name for logging/debugging."""
        return self.__class__.__name__


def event_listener(*event_types: str):
    """
    Decorator for registering event handler functions.

    Usage:
        @event_listener("memory.created", "memory.updated")
        async def on_memory_event(event: MemoryEvent):
            pass

    Args:
        *event_types: Event type identifiers to listen for

    Returns:
        Decorator that wraps the function as an EventHandler
    """

    def decorator(func: HandlerFunc) -> EventHandler:
        class FunctionHandler(EventHandler):
            async def handle(self, event: BaseEvent) -> None:
                await func(event)

            @property
            def event_types(self) -> list[str]:
                return list(event_types)

            @property
            def handler_name(self) -> str:
                return func.__name__

        return FunctionHandler()

    return decorator


__all__ = [
    "BaseEvent",
    "EventHandler",
    "EventT",
    "HandlerFunc",
    "event_listener",
]
