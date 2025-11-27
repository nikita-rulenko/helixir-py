"""
EventBus - Central event dispatcher and coordination point.

Responsibilities:
- Register/unregister event handlers
- Dispatch events to appropriate handlers
- Handle errors and retry logic
- Integrate with Float tracking for observability

Architecture:
- In-memory implementation (extensible to Redis/RabbitMQ later)
- Async event processing (Python 3.14 no-GIL!)
- Type-safe event routing
- Graceful error handling
"""

import asyncio
from collections import defaultdict
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseEvent, EventHandler

logger = logging.getLogger(__name__)


class EventBus:
    """
    Central event dispatcher for the application.

    Manages event handler registration and dispatches events
    to appropriate handlers asynchronously.

    Features:
    - Type-safe event routing
    - Async parallel handler execution
    - Error isolation (one handler failure doesn't affect others)
    - Observable via Float integration
    - Extensible (can add Redis/RabbitMQ backend later)

    Usage:
        bus = EventBus()
        bus.register(MyHandler())
        await bus.emit(MyEvent(...))
    """

    def __init__(self, enable_floats: bool = True):
        """
        Initialize EventBus.

        Args:
            enable_floats: Whether to emit Float tracking events
        """
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._enable_floats = enable_floats
        self._float_controller = None

    def register(self, handler: EventHandler) -> None:
        """
        Register an event handler.

        Args:
            handler: EventHandler instance to register
        """
        for event_type in handler.event_types:
            self._handlers[event_type].append(handler)
            logger.info(f"Registered handler {handler.handler_name} for event type '{event_type}'")

    def unregister(self, handler: EventHandler) -> None:
        """
        Unregister an event handler.

        Args:
            handler: EventHandler instance to unregister
        """
        for event_type in handler.event_types:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                logger.info(
                    f"Unregistered handler {handler.handler_name} from event type '{event_type}'"
                )

    async def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to all registered handlers.

        Handlers are invoked asynchronously in parallel.
        Handler errors are logged but don't stop other handlers.

        Args:
            event: Event instance to emit
        """
        event_type = event.event_type
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug(f"No handlers registered for event type '{event_type}'")
            return

        self._track_float("event.emitted", event)

        tasks = [self._execute_handler(handler, event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.warning(
                f"Event {event_type} ({event.event_id}): "
                f"{len(errors)}/{len(handlers)} handlers failed"
            )
            for error in errors:
                logger.error(f"Handler error: {error}", exc_info=error)

        self._track_float(
            "event.processed", event, handler_count=len(handlers), error_count=len(errors)
        )

    async def emit_many(self, events: list[BaseEvent]) -> None:
        """
        Emit multiple events in parallel.

        Useful for batch operations (e.g., emitting ChunkCreatedEvent for all chunks).

        Args:
            events: List of events to emit
        """
        tasks = [self.emit(event) for event in events]
        await asyncio.gather(*tasks)

    async def _execute_handler(self, handler: EventHandler, event: BaseEvent) -> None:
        """
        Execute a single handler with error handling.

        Args:
            handler: Handler to execute
            event: Event to pass to handler

        Raises:
            Exception: Handler exceptions are propagated for gather()
        """
        try:
            self._track_float("handler.started", event, handler=handler.handler_name)
            await handler.handle(event)
            self._track_float("handler.completed", event, handler=handler.handler_name)
        except Exception as e:
            self._track_float("handler.failed", event, handler=handler.handler_name, error=str(e))
            raise

    def _track_float(self, stage: str, event: BaseEvent, **extra: Any) -> None:
        """
        Emit Float tracking event.

        Args:
            stage: Pipeline stage (e.g., 'event.emitted', 'handler.completed')
            event: Event being processed
            **extra: Additional context for Float
        """
        if not self._enable_floats:
            return

        if self._float_controller is None:
            try:
                from helixir.toolkit.misc_toolbox.float_controller import float_event

                self._float_event = float_event
            except ImportError:
                logger.warning("Float tracking disabled: float_controller not available")
                self._enable_floats = False
                return

        if hasattr(self, "_float_event"):
            self._float_event(
                f"event_bus.{stage}",
                event_type=event.event_type,
                event_id=str(event.event_id),
                **extra,
            )

    def get_handler_count(self, event_type: str) -> int:
        """Get number of registered handlers for an event type."""
        return len(self._handlers.get(event_type, []))

    def get_stats(self) -> dict[str, Any]:
        """Get EventBus statistics for monitoring."""
        return {
            "total_event_types": len(self._handlers),
            "total_handlers": sum(len(handlers) for handlers in self._handlers.values()),
            "handlers_by_type": {
                event_type: len(handlers) for event_type, handlers in self._handlers.items()
            },
        }


_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Get the global EventBus instance.

    Creates a singleton instance on first call.
    Can be overridden for testing via set_event_bus().
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def set_event_bus(bus: EventBus) -> None:
    """
    Set the global EventBus instance.

    Useful for testing or custom configurations.

    Args:
        bus: EventBus instance to use globally
    """
    global _global_bus
    _global_bus = bus


__all__ = [
    "EventBus",
    "get_event_bus",
    "set_event_bus",
]
