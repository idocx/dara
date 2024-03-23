"""Search node data model."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from dara.result import RefinementResult


class SearchNodeData(BaseModel):
    current_result: Optional[RefinementResult]
    current_phases: list[Path]

    group_id: int = Field(default=-1, ge=-1)
    fom: float = Field(default=0, ge=0)
    lattice_strain: float = Field(default=0)

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

    isolated_missing_peaks: Optional[np.ndarray] = None
    isolated_extra_peaks: Optional[np.ndarray] = None

    peak_matcher_scores: Optional[dict[Path, float]] = None
    peak_matcher_score_threshold: Optional[float] = None


class SearchResult(BaseModel):
    refinement_result: RefinementResult
    phases: tuple[tuple[Path, ...], ...]
    foms: tuple[tuple[float, ...], ...]
    lattice_strains: tuple[tuple[float, ...], ...]
