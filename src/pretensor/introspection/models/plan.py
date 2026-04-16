from __future__ import annotations

from enum import StrEnum
from io import StringIO
from typing import Any

from pydantic import Field
from ruamel.yaml import YAML

from .base import PretensorModel


class ChartType(StrEnum):
    LINE = "line"
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    AREA = "area"
    PIE = "pie"
    BIG_VALUE = "big_value"
    TABLE = "table"


class SectionLayout(StrEnum):
    KPI_ROW = "kpi_row"
    SINGLE_CHART = "single_chart"
    TWO_COLUMN = "two_column"


class ChartConfig(PretensorModel):
    chart_type: ChartType
    metrics: list[str]
    dimensions: list[str] = Field(default_factory=list)
    time_range: str | None = None
    granularity: str | None = None
    sort: str | None = None
    limit: int | None = None
    format_overrides: dict[str, Any] = Field(default_factory=dict)


class Section(PretensorModel):
    title: str
    description: str | None = None
    layout: SectionLayout = SectionLayout.SINGLE_CHART
    charts: list[ChartConfig] = Field(default_factory=list)


class DashboardPlan(PretensorModel):
    title: str
    description: str | None = None
    default_time_range: str | None = None
    default_comparison: str | None = None
    sections: list[Section] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    target_page_path: str | None = None

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        yaml = YAML()
        yaml.default_flow_style = False
        buf = StringIO()
        yaml.dump(data, buf)
        return buf.getvalue()

    @classmethod
    def from_yaml(cls, yaml_str: str) -> DashboardPlan:
        yaml = YAML()
        data = yaml.load(yaml_str)
        return cls.model_validate(data)

    def to_markdown(self) -> str:
        """Render a human-readable markdown summary for review."""
        lines: list[str] = []
        lines.append(f"# {self.title}")
        if self.description:
            lines.append(f"\n{self.description}")
        lines.append("")

        meta: list[str] = []
        if self.default_time_range:
            meta.append(f"- **Time range:** {self.default_time_range}")
        if self.default_comparison:
            meta.append(f"- **Comparison:** {self.default_comparison}")
        if self.filters:
            meta.append(f"- **Filters:** {', '.join(self.filters)}")
        if self.target_page_path:
            meta.append(f"- **Page path:** `{self.target_page_path}`")
        if meta:
            lines.extend(meta)
            lines.append("")

        for idx, section in enumerate(self.sections, 1):
            lines.append(f"## {idx}. {section.title}")
            if section.description:
                lines.append(f"\n{section.description}")
            lines.append(f"\n*Layout: {section.layout.value}*\n")

            for chart in section.charts:
                lines.append(
                    f"- **{chart.chart_type.value}**: {', '.join(chart.metrics)}"
                )
                details: list[str] = []
                if chart.dimensions:
                    details.append(f"dimensions: {', '.join(chart.dimensions)}")
                if chart.granularity:
                    details.append(f"granularity: {chart.granularity}")
                if chart.time_range:
                    details.append(f"time range: {chart.time_range}")
                if chart.sort:
                    details.append(f"sort: {chart.sort}")
                if chart.limit:
                    details.append(f"limit: {chart.limit}")
                if details:
                    lines.append(f"  {' | '.join(details)}")
            lines.append("")

        return "\n".join(lines)
