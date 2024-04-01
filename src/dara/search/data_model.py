"""Search node data model."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

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

    isolated_missing_peaks: Optional[list[list[float]]] = None
    isolated_extra_peaks: Optional[list[list[float]]] = None

    @property
    def pretty_output(self):
        is_root_node = self.current_result is None and self.status == "expanded"

        if is_root_node:
            phase_string = "Root "
            if len(self.current_phases) > 1:
                phase_string += f"({', '.join([p.stem for p in self.current_phases])}) "
            return phase_string
        else:
            import colorful as cf

            status_color = {
                "max_depth": cf.blue,
                "error": cf.red,
                "expanded": cf.green,
            }
            status_str = status_color.get(self.status, lambda x: x)(self.status)
            phase_string = f"({status_str}) "
            phase_string += self.current_phases[-1].stem
            total_len = 3 + len(self.status) + len(self.current_phases[-1].stem)

            phase_string += " " * max(60 - total_len, 0)

        return f"{phase_string}" + (
            f"Rwp: {self.current_result.lst_data.rwp if self.current_result is not None else 1 * 100:.2f}% | "
            + f"Strain: {round(self.lattice_strain * 100, 2)}% | "
            + f"Group {self.group_id}  "
            if self.current_result is not None
            else ""
        )


class SearchResult(BaseModel):
    refinement_result: RefinementResult
    phases: tuple[tuple[Path, ...], ...]
    foms: tuple[tuple[float, ...], ...]
    lattice_strains: tuple[tuple[float, ...], ...]
