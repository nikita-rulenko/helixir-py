"""
Velocity Controller - Project Development Velocity Metrics.

Tracks:
- Bug resolution time (detection → fix)
- Feature implementation time (idea → deployment)
- Commit frequency & scope
- Memory evolution rate (insights/hour)
- Code quality metrics

Architecture:
    Event Stream → VelocityController → Metrics Aggregation → HelixDB Storage

Performance targets:
- Real-time event tracking (< 10ms overhead)
- Historical analysis (last 7/30/90 days)
- Predictive velocity forecasting
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helixir.core.client import HelixDBClient

logger = logging.getLogger(__name__)


class IssueStatus(str, Enum):
    """Issue lifecycle states."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    RESOLVED = "resolved"
    DEPRECATED = "deprecated"


class EventType(str, Enum):
    """Velocity tracking event types."""

    ISSUE_CREATED = "issue_created"
    ISSUE_STATUS_CHANGED = "issue_status_changed"
    ISSUE_RESOLVED = "issue_resolved"
    COMMIT_MADE = "commit_made"
    MEMORY_ADDED = "memory_added"
    FEATURE_COMPLETED = "feature_completed"


@dataclass
class VelocityEvent:
    """Single velocity tracking event."""

    event_type: EventType
    timestamp: datetime
    entity_id: str
    metadata: dict[str, Any]
    user_id: str


@dataclass
class VelocityMetrics:
    """Aggregated velocity metrics."""

    avg_bug_resolution_time: timedelta
    bugs_resolved_count: int
    bugs_open_count: int

    avg_feature_implementation_time: timedelta
    features_completed_count: int

    commits_per_day: float
    memories_per_session: float

    bug_reopen_rate: float
    memory_update_rate: float

    velocity_score: float

    period_start: datetime
    period_end: datetime


class VelocityController:
    """
    Track and analyze project development velocity.

    Features:
    - Real-time event tracking
    - Historical metrics aggregation
    - Velocity score calculation
    - Predictive analytics (TODO)
    - HelixDB integration for persistence
    """

    def __init__(
        self,
        db_client: HelixDBClient | None = None,
        project_id: str = "helixir",
    ):
        """
        Initialize VelocityController.

        Args:
            db_client: HelixDB client for persistence
            project_id: Project identifier for multi-project support
        """
        self.db_client = db_client
        self.project_id = project_id

        self._events: list[VelocityEvent] = []

        self._issue_states: dict[str, dict[str, Any]] = {}

        logger.info("VelocityController initialized for project: %s", project_id)

    async def track_event(self, event: VelocityEvent) -> None:
        """
        Track single velocity event.

        Args:
            event: Velocity event to track
        """
        self._events.append(event)

        if event.event_type == EventType.ISSUE_CREATED:
            self._issue_states[event.entity_id] = {
                "status": IssueStatus.CREATED,
                "created_at": event.timestamp,
                "transitions": [{"from": None, "to": IssueStatus.CREATED, "at": event.timestamp}],
                "metadata": event.metadata,
            }

        elif event.event_type == EventType.ISSUE_STATUS_CHANGED:
            if event.entity_id in self._issue_states:
                old_status = self._issue_states[event.entity_id]["status"]
                new_status = event.metadata.get("new_status")

                self._issue_states[event.entity_id]["status"] = new_status
                self._issue_states[event.entity_id]["transitions"].append(
                    {"from": old_status, "to": new_status, "at": event.timestamp}
                )

        elif event.event_type == EventType.ISSUE_RESOLVED:
            if event.entity_id in self._issue_states:
                self._issue_states[event.entity_id]["status"] = IssueStatus.RESOLVED
                self._issue_states[event.entity_id]["resolved_at"] = event.timestamp
                self._issue_states[event.entity_id]["transitions"].append(
                    {
                        "from": self._issue_states[event.entity_id]["status"],
                        "to": IssueStatus.RESOLVED,
                        "at": event.timestamp,
                    }
                )

        if self.db_client:
            await self._persist_event(event)

        logger.debug("Tracked event: %s for entity: %s", event.event_type, event.entity_id)

    async def calculate_metrics(
        self,
        period_days: int = 7,
    ) -> VelocityMetrics:
        """
        Calculate velocity metrics for specified period.

        Args:
            period_days: Time window for metrics calculation

        Returns:
            VelocityMetrics object with aggregated data
        """
        now = datetime.now(UTC)
        period_start = now - timedelta(days=period_days)

        period_events = [e for e in self._events if e.timestamp >= period_start]

        resolved_bugs = [e for e in period_events if e.event_type == EventType.ISSUE_RESOLVED]
        bug_resolution_times = []

        for resolved_event in resolved_bugs:
            issue_id = resolved_event.entity_id
            if issue_id in self._issue_states:
                created_at = self._issue_states[issue_id]["created_at"]
                resolved_at = self._issue_states[issue_id]["resolved_at"]
                resolution_time = resolved_at - created_at
                bug_resolution_times.append(resolution_time)

        avg_bug_resolution = (
            sum(bug_resolution_times, timedelta()) / len(bug_resolution_times)
            if bug_resolution_times
            else timedelta(0)
        )

        open_bugs = [
            iid
            for iid, state in self._issue_states.items()
            if state["status"] not in [IssueStatus.RESOLVED, IssueStatus.DEPRECATED]
        ]

        commits = [e for e in period_events if e.event_type == EventType.COMMIT_MADE]
        commits_per_day = len(commits) / period_days if period_days > 0 else 0.0

        memories = [e for e in period_events if e.event_type == EventType.MEMORY_ADDED]
        memories_per_session = len(memories) / max(1, len(commits))

        features = [e for e in period_events if e.event_type == EventType.FEATURE_COMPLETED]

        velocity_score = self._calculate_velocity_score(
            avg_resolution_time=avg_bug_resolution,
            commits_per_day=commits_per_day,
            bugs_resolved=len(resolved_bugs),
            features_completed=len(features),
        )

        return VelocityMetrics(
            avg_bug_resolution_time=avg_bug_resolution,
            bugs_resolved_count=len(resolved_bugs),
            bugs_open_count=len(open_bugs),
            avg_feature_implementation_time=timedelta(0),
            features_completed_count=len(features),
            commits_per_day=commits_per_day,
            memories_per_session=memories_per_session,
            bug_reopen_rate=0.0,
            memory_update_rate=0.0,
            velocity_score=velocity_score,
            period_start=period_start,
            period_end=now,
        )

    def _calculate_velocity_score(
        self,
        avg_resolution_time: timedelta,
        commits_per_day: float,
        bugs_resolved: int,
        features_completed: int,
    ) -> float:
        """
        Calculate composite velocity score (0-100).

        Formula:
        - Fast bug resolution: +40 points
        - High commit frequency: +30 points
        - Bug resolution count: +15 points
        - Feature completion: +15 points

        Args:
            avg_resolution_time: Average bug resolution time
            commits_per_day: Commits per day
            bugs_resolved: Bugs resolved count
            features_completed: Features completed count

        Returns:
            Velocity score (0-100)
        """
        score = 0.0

        resolution_hours = avg_resolution_time.total_seconds() / 3600
        if resolution_hours < 24:
            score += 40.0
        elif resolution_hours < 168:
            score += 40.0 * (1 - (resolution_hours - 24) / (168 - 24))

        score += min(30.0, commits_per_day * 6)

        score += min(15.0, bugs_resolved * 3)

        score += min(15.0, features_completed * 5)

        return min(100.0, score)

    async def get_issue_lifecycle(self, issue_id: str) -> dict[str, Any] | None:
        """
        Get issue lifecycle history.

        Args:
            issue_id: Issue identifier

        Returns:
            Issue state with transitions, or None if not found
        """
        return self._issue_states.get(issue_id)

    async def _persist_event(self, event: VelocityEvent) -> None:
        """
        Persist event to HelixDB.

        Args:
            event: Event to persist

        TODO: Implement HelixDB schema:
        - VelocityEvent nodes
        - TRACKED_BY edges to Project node
        - Time-series queries
        """

    def get_stats(self) -> dict[str, Any]:
        """Get controller statistics."""
        return {
            "total_events": len(self._events),
            "tracked_issues": len(self._issue_states),
            "open_issues": len(
                [
                    i
                    for i in self._issue_states.values()
                    if i["status"] not in [IssueStatus.RESOLVED, IssueStatus.DEPRECATED]
                ]
            ),
        }
