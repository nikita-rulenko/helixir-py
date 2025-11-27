"""Reasoning models for representing logical relationships."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

ReasoningType = Literal[
    "IMPLIES", "BECAUSE", "CONTRADICTS", "SUPERSEDES", "DERIVED_FROM", "SUPPORTS", "REFUTES"
]


class ReasoningRelation(BaseModel):
    """
    Represents a reasoning relationship between two memories.

    Defines logical connections like:
    - IMPLIES: A implies B
    - BECAUSE: A is true because of B
    - CONTRADICTS: A contradicts B
    - SUPERSEDES: A replaces/updates B
    - DERIVED_FROM: A is derived from B
    """

    relation_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique relation ID")
    from_memory_id: str = Field(..., description="Source memory ID")
    to_memory_id: str = Field(..., description="Target memory ID")
    relation_type: ReasoningType = Field(..., description="Type of reasoning relation")

    strength: int = Field(default=50, ge=0, le=100, description="Strength of the relation (0-100)")
    confidence: int = Field(
        default=50, ge=0, le=100, description="Confidence in the relation (0-100)"
    )
    explanation: str | None = Field(default=None, description="Human-readable explanation")

    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    created_by: str | None = Field(default=None, description="Creator (user_id or 'system')")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "relation_id": "rel-123",
                "from_memory_id": "mem-456",
                "to_memory_id": "mem-789",
                "relation_type": "IMPLIES",
                "strength": 80,
                "confidence": 90,
                "explanation": "Learning Python implies ability to write code",
                "created_by": "system",
            }
        }
    )


class ReasoningChain(BaseModel):
    """
    Represents a chain of reasoning (sequence of related memories).

    Example: A → B → C → D
    """

    chain_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chain ID")
    memory_ids: list[str] = Field(..., description="Ordered list of memory IDs in chain")
    relation_types: list[ReasoningType] = Field(
        ..., description="Relations between consecutive memories"
    )
    total_strength: float = Field(default=0.0, description="Combined strength of the chain")
    length: int = Field(default=0, description="Number of memories in chain")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chain_id": "chain-abc",
                "memory_ids": ["mem-1", "mem-2", "mem-3"],
                "relation_types": ["IMPLIES", "BECAUSE"],
                "total_strength": 0.75,
                "length": 3,
            }
        }
    )


class ConflictInfo(BaseModel):
    """Information about a memory conflict."""

    memory_a_id: str = Field(..., description="First conflicting memory ID")
    memory_b_id: str = Field(..., description="Second conflicting memory ID")
    conflict_type: str = Field(..., description="Type of conflict (contradiction, temporal, etc.)")
    severity: int = Field(default=50, ge=0, le=100, description="Severity of conflict (0-100)")
    resolution: str | None = Field(default=None, description="Suggested resolution strategy")
    detected_at: datetime = Field(
        default_factory=datetime.now, description="When conflict was detected"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "memory_a_id": "mem-123",
                "memory_b_id": "mem-456",
                "conflict_type": "contradiction",
                "severity": 80,
                "resolution": "prefer_newer",
                "detected_at": "2025-11-06T20:00:00",
            }
        }
    )
