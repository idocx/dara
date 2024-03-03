"""Search node data model."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from dara.result import RefinementResult


class SearchNodeData(BaseModel):
    current_result: Optional[RefinementResult]
    current_phases: list[Path]

    group_id: int = Field(default=0, ge=0)
    fom: float = Field(default=0, ge=0)
    status: Literal[
        "pending",
        "max_depth",
        "error",
        "no_improvement",
        "running",
        "expanded",
        "similar_structure",
        "low_weight_fraction",
        "duplicate",
    ] = "pending"

    peak_matcher_scores: Optional[dict[Path, float]] = None
    peak_matcher_score_threshold: Optional[float] = None
