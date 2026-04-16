"""Shared base model for graph package Pydantic types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["GraphModel"]


class GraphModel(BaseModel):
    model_config = ConfigDict(frozen=True)
