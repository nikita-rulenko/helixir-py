"""Pydantic models for LLM extraction results."""

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_serializer

if TYPE_CHECKING:
    from datetime import datetime


class ExtractedMemory(BaseModel):
    """Represents a memory extracted from text by LLM."""

    content: str = Field(..., description="The memory content")
    memory_type: str = Field(..., description="Type: fact, preference, event, skill, goal, etc.")
    certainty: int = Field(50, ge=0, le=100, description="Confidence in extraction (0-100)")
    importance: int = Field(50, ge=0, le=100, description="Estimated importance (0-100)")
    concepts: list[str] = Field(default_factory=list, description="Related ontological concepts")
    context: dict[str, bool] = Field(
        default_factory=dict, description="Context tags (work, personal, etc.)"
    )
    valid_from: datetime | None = Field(None, description="Temporal validity start")
    valid_until: datetime | None = Field(None, description="Temporal validity end")


class ExtractedEntity(BaseModel):
    """Represents an entity extracted from text."""

    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Type: Person, Organization, Location, Object")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional properties (can be str, int, list, dict)"
    )

    @field_serializer("properties")
    def serialize_properties(self, value: dict[str, Any]) -> str:
        """
        Serialize properties to JSON string for HelixDB storage.

        HelixDB schema expects properties as String type.
        LLM may return int, list, dict values - we serialize to JSON.
        """
        return json.dumps(value) if value else "{}"


class ExtractedRelation(BaseModel):
    """Represents a reasoning relation extracted from text."""

    from_memory_content: str = Field(..., description="Source memory content")
    to_memory_content: str = Field(..., description="Target memory content")
    relation_type: str = Field(..., description="IMPLIES, BECAUSE, CONTRADICTS, etc.")
    strength: int = Field(50, ge=0, le=100, description="Relation strength (0-100)")
    confidence: int = Field(50, ge=0, le=100, description="Confidence in relation (0-100)")
    explanation: str | None = Field(None, description="Why this relation exists")


class ExtractionResult(BaseModel):
    """Complete result of LLM extraction."""

    memories: list[ExtractedMemory] = Field(default_factory=list, description="Extracted memories")
    entities: list[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    relations: list[ExtractedRelation] = Field(
        default_factory=list, description="Extracted relations"
    )
    summary: str | None = Field(None, description="Optional summary of the extraction")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Extraction metadata (model, tokens, etc.)"
    )
