"""Validation report models — shared across all validation layers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from .base import PretensorModel

__all__ = [
    "Severity",
    "RepairState",
    "ValidationIssue",
    "ComponentScore",
    "LayerResult",
    "AnalyticalQualityResult",
    "ValidationReport",
    "LayerStatus",
]

LayerStatus = Literal["pass", "fail", "repaired", "not_evaluated"]


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RepairState(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    REPAIRED = "repaired"
    NOT_EVALUATED = "not_evaluated"


class ValidationIssue(PretensorModel):
    model_config = PretensorModel.model_config.copy()
    model_config["frozen"] = False

    severity: Severity
    layer: str
    path: str
    message: str


class LayerResult(PretensorModel):
    model_config = PretensorModel.model_config.copy()
    model_config["frozen"] = False

    layer_name: str
    status: LayerStatus
    repair_state: RepairState = RepairState.PASS
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)


class ComponentScore(PretensorModel):
    """Per-chart quality score produced by the L4 analyst review."""

    model_config = PretensorModel.model_config.copy()
    model_config["frozen"] = False

    score: int
    answers_main_question: bool
    priority_rank: int
    current_position: int
    note: str
    subtracting: str | None = None
    suggested_action: str | None = None


class AnalyticalQualityResult(LayerResult):
    """Extended LayerResult for L4, carrying per-chart ComponentScore breakdown."""

    model_config = LayerResult.model_config.copy()

    overall_score: int = 0
    overall_summary: str = ""
    question_answered: bool = False
    components: dict[str, ComponentScore] = Field(default_factory=dict)
    repair_tier: int = 1


class ValidationReport(PretensorModel):
    model_config = PretensorModel.model_config.copy()
    model_config["frozen"] = False

    dashboard_name: str
    layers: list[LayerResult] = Field(default_factory=list)
    repair_actions_taken: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.status == "pass" for r in self.layers)

    @property
    def all_issues(self) -> list[ValidationIssue]:
        return [issue for layer in self.layers for issue in layer.issues]

    @property
    def error_count(self) -> int:
        return sum(r.error_count for r in self.layers)

    @property
    def warning_count(self) -> int:
        return sum(r.warning_count for r in self.layers)
