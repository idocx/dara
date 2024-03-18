"""Search node data model."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from dara.result import RefinementResult
    from dara.search.tree import BaseSearchTree
    from treelib import Node


class SearchNodeData(BaseModel):
    current_result: Optional[RefinementResult]
    current_phases: list[Path]

    group_id: int = Field(default=-1, ge=-1)
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


class SearchResult(BaseModel):
    refinement_result: RefinementResult
    phases: tuple[tuple[Path, ...], ...]
    foms: tuple[tuple[float, ...], ...]

    @classmethod
    def from_search_node(
        cls, search_node: Node, search_tree: BaseSearchTree
    ) -> "SearchResult":
        phase_combinations = search_tree.get_phase_combinations(search_node)

        phases = tuple(
            tuple(phase for phase, fom in phases) for phases in phase_combinations
        )
        foms = tuple(
            tuple(fom for phase, fom in phases) for phases in phase_combinations
        )

        results = search_node.data.current_result
        if results is None:
            raise ValueError("Search node has no result")

        return cls(
            refinement_result=results,
            phases=phases,
            foms=foms,
        )
