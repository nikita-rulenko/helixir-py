"""
Schema-Query Sync Pipeline Events.

Event-driven architecture for automated schema-query synchronization,
validation, and fixing.
"""

from dataclasses import dataclass

from helixir.core.events.base import BaseEvent


@dataclass(frozen=True)
class SchemaValidationStartedEvent(BaseEvent):
    """Emitted when schema validation begins."""

    schema_file: str

    @property
    def event_type(self) -> str:
        return "schema.validation.started"


@dataclass(frozen=True)
class SchemaValidationCompletedEvent(BaseEvent):
    """Emitted when schema validation completes."""

    @property
    def event_type(self) -> str:
        return "schema.validation.completed"

    schema_file: str
    is_valid: bool
    errors: list[str]
    stats: dict[str, int]


@dataclass(frozen=True)
class QueryValidationStartedEvent(BaseEvent):
    """Emitted when query validation begins."""

    @property
    def event_type(self) -> str:
        return "query.validation.started"

    queries_file: str


@dataclass(frozen=True)
class QueryValidationCompletedEvent(BaseEvent):
    """Emitted when query validation completes."""

    @property
    def event_type(self) -> str:
        return "query.validation.completed"

    queries_file: str
    total_queries: int
    valid_queries: int
    errors: list[str]


@dataclass(frozen=True)
class SyncCheckStartedEvent(BaseEvent):
    """Emitted when schema-query sync check begins."""

    @property
    def event_type(self) -> str:
        return "sync.check.started"

    schema_file: str
    queries_file: str


@dataclass(frozen=True)
class SyncIssueDetectedEvent(BaseEvent):
    """Emitted when a schema-query sync issue is detected."""

    @property
    def event_type(self) -> str:
        return "sync.issue.detected"

    query_name: str
    issue_type: str
    details: dict[str, any]
    auto_fixable: bool
    suggested_fix: str | None = None


@dataclass(frozen=True)
class SyncCheckCompletedEvent(BaseEvent):
    """Emitted when sync check completes."""

    @property
    def event_type(self) -> str:
        return "sync.check.completed"

    total_issues: int
    auto_fixable_issues: int
    manual_issues: int
    issues_by_query: dict[str, list[dict]]


@dataclass(frozen=True)
class FixStartedEvent(BaseEvent):
    """Emitted when auto-fix process starts."""

    @property
    def event_type(self) -> str:
        return "fix.started"

    total_fixes: int


@dataclass(frozen=True)
class FixAppliedEvent(BaseEvent):
    """Emitted when a fix is successfully applied."""

    @property
    def event_type(self) -> str:
        return "fix.applied"

    query_name: str
    fix_type: str
    details: dict[str, any]


@dataclass(frozen=True)
class FixFailedEvent(BaseEvent):
    """Emitted when a fix attempt fails."""

    @property
    def event_type(self) -> str:
        return "fix.failed"

    query_name: str
    fix_type: str
    error: str


@dataclass(frozen=True)
class FixCompletedEvent(BaseEvent):
    """Emitted when all fixes are applied."""

    @property
    def event_type(self) -> str:
        return "fix.completed"

    successful_fixes: int
    failed_fixes: int
    remaining_issues: int


@dataclass(frozen=True)
class RegenerationStartedEvent(BaseEvent):
    """Emitted when query regeneration starts."""

    @property
    def event_type(self) -> str:
        return "regeneration.started"

    output_file: str


@dataclass(frozen=True)
class RegenerationCompletedEvent(BaseEvent):
    """Emitted when regeneration completes."""

    @property
    def event_type(self) -> str:
        return "regeneration.completed"

    output_file: str
    queries_regenerated: int


@dataclass(frozen=True)
class PipelineStartedEvent(BaseEvent):
    """Emitted when sync pipeline starts."""

    @property
    def event_type(self) -> str:
        return "pipeline.started"

    schema_file: str
    queries_file: str
    auto_fix: bool
    dry_run: bool


@dataclass(frozen=True)
class PipelineCompletedEvent(BaseEvent):
    """Emitted when pipeline completes successfully."""

    @property
    def event_type(self) -> str:
        return "pipeline.completed"

    total_issues_found: int
    issues_fixed: int
    issues_remaining: int
    queries_updated: int
    ready_for_deployment: bool


@dataclass(frozen=True)
class PipelineFailedEvent(BaseEvent):
    """Emitted when pipeline fails."""

    @property
    def event_type(self) -> str:
        return "pipeline.failed"

    stage: str
    error: str
