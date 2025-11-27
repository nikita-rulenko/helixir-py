"""
Event System - Event-driven architecture for Helixir.

Core Components:
- BaseEvent: Immutable event records
- EventBus: Central dispatcher
- EventHandler: Event processors
- event_listener: Handler decorator

Design Philosophy:
- Events = facts (immutable, timestamped)
- Handlers = reactions (async, isolated)
- Bus = coordinator (routing, error handling)
- Observable = Float integration automatic

Quick Start:
    from helixir.core.events import BaseEvent, get_event_bus, event_listener

    @dataclass(frozen=True)
    class MyEvent(BaseEvent):
        data: str

        @property
        def event_type(self) -> str:
            return "my.event"

    @event_listener("my.event")
    async def on_my_event(event: MyEvent):
        print(f"Received: {event.data}")

    bus = get_event_bus()
    bus.register(on_my_event)

    await bus.emit(MyEvent(data="Hello!"))
"""

from .base import BaseEvent, EventHandler, EventT, HandlerFunc, event_listener
from .bus import EventBus, get_event_bus, set_event_bus
from .chunking import (
    ChunkChainedEvent,
    ChunkCreatedEvent,
    ChunkingCompleteEvent,
    ChunkingFailedEvent,
    ChunkingStartedEvent,
    ChunkLinkedEvent,
    MemoryCreatedEvent,
)

__all__ = [
    "BaseEvent",
    "ChunkChainedEvent",
    "ChunkCreatedEvent",
    "ChunkLinkedEvent",
    "ChunkingCompleteEvent",
    "ChunkingFailedEvent",
    "ChunkingStartedEvent",
    "EventBus",
    "EventHandler",
    "EventT",
    "HandlerFunc",
    "MemoryCreatedEvent",
    "event_listener",
    "get_event_bus",
    "set_event_bus",
]
