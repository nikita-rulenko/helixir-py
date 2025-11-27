"""Pydantic models for memory entities."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EntityType(str, Enum):
    """
    Entity types for tech domain classification.

    Categories:
    - Physical: Person, Organization, Location
    - Digital: System, Component, Resource, Concept
    - Process: Process, Event
    """

    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"

    SYSTEM = "System"
    COMPONENT = "Component"
    RESOURCE = "Resource"
    CONCEPT = "Concept"

    PROCESS = "Process"
    EVENT = "Event"


class Memory(BaseModel):
    """
    Represents a memory node in the graph.

    Memories store facts, preferences, experiences, and other information
    with metadata for temporal validity, importance, and context.
    """

    memory_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique memory ID")
    content: str = Field(..., description="The actual memory content")
    memory_type: str = Field(
        ..., description="Type of memory: fact, preference, event, skill, etc."
    )
    user_id: str = Field(..., description="ID of the user this memory belongs to")

    certainty: int = Field(default=100, ge=0, le=100, description="Confidence level (0-100)")
    importance: int = Field(default=50, ge=0, le=100, description="Importance score (0-100)")

    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    valid_from: datetime = Field(
        default_factory=datetime.now, description="Start of validity period"
    )
    valid_until: datetime | None = Field(
        default=None, description="End of validity period (None = forever)"
    )

    context: dict[str, Any] = Field(
        default_factory=dict, description="Contextual metadata (work, personal, etc.)"
    )

    context_tags: str = Field(default="", description="Comma-separated context tags for grouping")
    source: str = Field(default="manual", description="Source of memory: manual, llm, import, api")
    metadata: str = Field(default="{}", description="JSON string for flexible metadata storage")

    concepts: list[str] = Field(
        default_factory=list, description="List of concept IDs this memory relates to"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "memory_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "I love Python programming",
                "memory_type": "preference",
                "user_id": "user123",
                "certainty": 90,
                "importance": 70,
                "context": {"work": True, "personal": False},
                "concepts": ["Preference", "Skill"],
            }
        }
    )


class Entity(BaseModel):
    """
    Represents an entity (person, organization, system, process, etc).

    Entities are extracted from memories and linked to them.
    Supports 9 entity types across Physical/Digital/Process categories.
    """

    entity_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entity ID")
    name: str = Field(..., description="Entity name")
    entity_type: EntityType = Field(..., description="Entity type (9 types across 3 categories)")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties with subtype/technology/domain metadata",
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_id": "550e8400-e29b-41d4-a716-446655440001",
                "name": "HelixDB",
                "entity_type": "System",
                "properties": {
                    "subtype": "database",
                    "technology": "rust",
                    "category": "graph-vector-db",
                },
            }
        }
    )


class Context(BaseModel):
    """
    Represents a context for memory validity.

    Contexts define when and where memories are applicable.
    """

    context_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique context ID")
    name: str = Field(..., description="Context name (Work, Home, Travel, etc.)")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Context-specific properties"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "context_id": "550e8400-e29b-41d4-a716-446655440002",
                "name": "Work",
                "properties": {"location": "office", "hours": "9-5"},
            }
        }
    )


class MemoryStats(BaseModel):
    """Statistics about memories."""

    total_memories: int = Field(default=0, description="Total number of memories")
    memories_by_type: dict[str, int] = Field(
        default_factory=dict, description="Count by memory type"
    )
    memories_by_user: dict[str, int] = Field(default_factory=dict, description="Count by user")
    avg_certainty: float = Field(default=0.0, description="Average certainty score")
    avg_importance: float = Field(default=0.0, description="Average importance score")
    oldest_memory: datetime | None = Field(default=None, description="Oldest memory timestamp")
    newest_memory: datetime | None = Field(default=None, description="Newest memory timestamp")
