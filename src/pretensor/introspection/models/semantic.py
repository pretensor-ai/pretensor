from __future__ import annotations

from enum import StrEnum
from io import StringIO
from typing import Literal

from pydantic import Field
from ruamel.yaml import YAML

from .base import PretensorModel


class AttributeRole(StrEnum):
    IDENTIFIER = "identifier"
    DIMENSION = "dimension"
    MEASURE = "measure"
    TIME_DIMENSION = "time_dimension"


class MetricType(StrEnum):
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    SUM = "sum"
    AVERAGE = "average"
    DERIVED = "derived"


class FormatConfig(PretensorModel):
    """Per-attribute or per-metric format override."""

    format_type: Literal["number", "currency", "percentage", "date"] | None = None
    decimals: int | None = None
    symbol: str | None = None
    position: Literal["prefix", "suffix"] | None = None
    suffix: str | None = None
    thousands_separator: str | None = None
    format_string: str | None = None


class Attribute(PretensorModel):
    name: str
    description: str
    role: AttributeRole
    source_column: str
    format_config: FormatConfig | None = None


class Metric(PretensorModel):
    name: str
    description: str
    type: MetricType
    field: str | None = None
    expression: str | None = None
    time_dimension: str | None = None
    granularity: str | None = None
    format_config: FormatConfig | None = None
    filters: list[str] = Field(default_factory=list)


class Entity(PretensorModel):
    name: str
    description: str
    source_table: str
    attributes: list[Attribute] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)


class Domain(PretensorModel):
    name: str
    description: str
    entities: list[Entity] = Field(default_factory=list)


class NumberFormatDefaults(PretensorModel):
    decimals: int = 0
    thousands_separator: str = ","


class CurrencyFormatDefaults(PretensorModel):
    symbol: str = "$"
    position: Literal["prefix", "suffix"] = "prefix"
    decimals: int = 2


class PercentageFormatDefaults(PretensorModel):
    decimals: int = 1
    suffix: str = "%"


class DateFormatDefaults(PretensorModel):
    format_string: str = "YYYY-MM-DD"


class FormatDefaults(PretensorModel):
    number: NumberFormatDefaults = Field(default_factory=NumberFormatDefaults)
    currency: CurrencyFormatDefaults = Field(default_factory=CurrencyFormatDefaults)
    percentage: PercentageFormatDefaults = Field(
        default_factory=PercentageFormatDefaults
    )
    date: DateFormatDefaults = Field(default_factory=DateFormatDefaults)
    chart_preferences: dict[str, str] = Field(default_factory=dict)


class SemanticLayer(PretensorModel):
    connection_name: str
    domains: list[Domain] = Field(default_factory=list)
    format_defaults: FormatDefaults = Field(default_factory=FormatDefaults)

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        yaml = YAML()
        yaml.default_flow_style = False
        buf = StringIO()
        yaml.dump(data, buf)
        return buf.getvalue()

    @classmethod
    def from_yaml(cls, yaml_str: str) -> SemanticLayer:
        yaml = YAML()
        data = yaml.load(yaml_str)
        return cls.model_validate(data)
