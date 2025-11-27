"""Data models for ontology."""

from enum import Enum

from pydantic import BaseModel, Field


class ConceptType(str, Enum):
    """Type of concept."""

    ABSTRACT = "Abstract"
    CONCRETE = "Concrete"


class Concept(BaseModel):
    """
    Represents a concept in the ontology.

    A concept is a category or class that memories can be classified under.
    Concepts form a hierarchy (IS_A, HAS_SUBTYPE relationships).
    """

    concept_id: str = Field(..., description="Unique identifier for the concept")
    name: str = Field(..., description="Human-readable name")
    type: ConceptType = Field(..., description="Abstract or Concrete")
    description: str = Field("", description="Description of what this concept represents")
    parent_concept: str | None = Field(None, description="Parent concept ID (for hierarchy)")

    def __str__(self) -> str:
        """String representation."""
        return f"Concept({self.name}, type={self.type.value})"

    def __repr__(self) -> str:
        """Repr representation."""
        return f"Concept(id={self.concept_id!r}, name={self.name!r}, type={self.type.value})"


class ConceptRelation(BaseModel):
    """
    Represents a relationship between concepts.

    Types:
    - IS_A: Concept A is a type of Concept B
    - HAS_SUBTYPE: Concept A has subtype Concept B
    """

    from_concept: str = Field(..., description="Source concept ID")
    to_concept: str = Field(..., description="Target concept ID")
    relation_type: str = Field(..., description="Type of relation (IS_A, HAS_SUBTYPE)")

    def __str__(self) -> str:
        """String representation."""
        return f"{self.from_concept} --{self.relation_type}--> {self.to_concept}"


class OntologyStats(BaseModel):
    """Statistics about the ontology."""

    total_concepts: int = Field(0, description="Total number of concepts")
    total_relations: int = Field(0, description="Total number of relations")
    concepts_by_type: dict[str, int] = Field(
        default_factory=dict, description="Concepts grouped by type"
    )
    max_depth: int = Field(0, description="Maximum depth of concept hierarchy")
