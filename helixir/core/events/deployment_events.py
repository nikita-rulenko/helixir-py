"""
Deployment Events - Events for the deployment pipeline.

Event Flow:
1. DeploymentStartedEvent
2. PreparationStartedEvent
3. SchemaGeneratedEvent / QueriesGeneratedEvent
4. ValidationCompletedEvent
5. VersionBumpedEvent
6. HostCheckStartedEvent
7. HostCheckCompletedEvent
8. DeploymentStartedEvent (actual deploy)
9. SchemaDeployedEvent / QueriesDeployedEvent
10. VerificationStartedEvent
11. VerificationCompletedEvent
12. DeploymentCompletedEvent / DeploymentFailedEvent
13. RollbackStartedEvent (if failure)
14. RollbackCompletedEvent

All events are immutable (frozen=True) for safety.
"""

from dataclasses import dataclass

from helixir.core.events.base import BaseEvent


@dataclass(frozen=True)
class DeploymentStartedEvent(BaseEvent):
    """Deployment pipeline started."""

    pipeline_id: str
    dry_run: bool
    timestamp: str

    @property
    def event_type(self) -> str:
        return "deployment.started"


@dataclass(frozen=True)
class PreparationStartedEvent(BaseEvent):
    """Preparation stage started (schema/queries generation + validation)."""

    pipeline_id: str
    timestamp: str

    @property
    def event_type(self) -> str:
        return "preparation.started"


@dataclass(frozen=True)
class SchemaGeneratedEvent(BaseEvent):
    """Schema generated successfully."""

    pipeline_id: str
    schema_file: str
    lines: int
    timestamp: str

    @property
    def event_type(self) -> str:
        return "schema.generated"


@dataclass(frozen=True)
class QueriesGeneratedEvent(BaseEvent):
    """Queries generated successfully."""

    pipeline_id: str
    queries_file: str
    query_count: int
    timestamp: str

    @property
    def event_type(self) -> str:
        return "queries.generated"


@dataclass(frozen=True)
class ValidationCompletedEvent(BaseEvent):
    """Validation completed (schema + queries + sync check)."""

    pipeline_id: str
    schema_valid: bool
    queries_valid: bool
    sync_ok: bool
    issues_found: int
    issues_fixed: int
    timestamp: str

    @property
    def event_type(self) -> str:
        return "validation.completed"


@dataclass(frozen=True)
class VersionBumpedEvent(BaseEvent):
    """Version bumped."""

    pipeline_id: str
    target: str
    old_version: str
    new_version: str
    timestamp: str

    @property
    def event_type(self) -> str:
        return "version.bumped"


@dataclass(frozen=True)
class HostCheckStartedEvent(BaseEvent):
    """Host health check started."""

    pipeline_id: str
    host: str
    port: int
    timestamp: str

    @property
    def event_type(self) -> str:
        return "host_check.started"


@dataclass(frozen=True)
class HostCheckCompletedEvent(BaseEvent):
    """Host health check completed."""

    pipeline_id: str
    success: bool
    vm_reachable: bool
    helixdb_healthy: bool
    deployed_schema_version: str | None
    deployed_queries_version: str | None
    error: str | None
    timestamp: str

    @property
    def event_type(self) -> str:
        return "host_check.completed"


@dataclass(frozen=True)
class SchemaDeployedEvent(BaseEvent):
    """Schema deployed to HelixDB."""

    pipeline_id: str
    version: str
    file: str
    timestamp: str

    @property
    def event_type(self) -> str:
        return "schema.deployed"


@dataclass(frozen=True)
class QueriesDeployedEvent(BaseEvent):
    """Queries deployed to HelixDB."""

    pipeline_id: str
    version: str
    file: str
    query_count: int
    timestamp: str

    @property
    def event_type(self) -> str:
        return "queries.deployed"


@dataclass(frozen=True)
class VerificationStartedEvent(BaseEvent):
    """Post-deployment verification started."""

    pipeline_id: str
    expected_schema_version: str
    expected_queries_version: str
    timestamp: str

    @property
    def event_type(self) -> str:
        return "verification.started"


@dataclass(frozen=True)
class VerificationCompletedEvent(BaseEvent):
    """Post-deployment verification completed."""

    pipeline_id: str
    success: bool
    schema_version_ok: bool
    queries_version_ok: bool
    sample_queries_ok: bool
    performance_ok: bool
    errors: list[str]
    timestamp: str

    @property
    def event_type(self) -> str:
        return "verification.completed"


@dataclass(frozen=True)
class DeploymentCompletedEvent(BaseEvent):
    """Deployment pipeline completed successfully."""

    pipeline_id: str
    schema_version: str
    queries_version: str
    duration_seconds: float
    timestamp: str

    @property
    def event_type(self) -> str:
        return "deployment.completed"


@dataclass(frozen=True)
class DeploymentFailedEvent(BaseEvent):
    """Deployment pipeline failed."""

    pipeline_id: str
    stage: str
    error: str
    timestamp: str

    @property
    def event_type(self) -> str:
        return "deployment.failed"


@dataclass(frozen=True)
class RollbackStartedEvent(BaseEvent):
    """Rollback started due to deployment failure."""

    pipeline_id: str
    reason: str
    target_schema_version: str | None
    target_queries_version: str | None
    timestamp: str

    @property
    def event_type(self) -> str:
        return "rollback.started"


@dataclass(frozen=True)
class RollbackCompletedEvent(BaseEvent):
    """Rollback completed."""

    pipeline_id: str
    success: bool
    schema_restored: bool
    queries_restored: bool
    error: str | None
    timestamp: str

    @property
    def event_type(self) -> str:
        return "rollback.completed"
