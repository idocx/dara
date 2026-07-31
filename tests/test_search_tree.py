import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from treelib import Node

from dara.refine import RefinementPhase
from dara.result import DiaResult, LstResult, RefinementMetrics, RefinementResult
from dara.search.data_model import SearchNodeData
from dara.search.tree import (
    MATERIAL_RWP_IMPROVEMENT,
    BaseSearchTree,
    keep_branch_despite_intensity_order,
)


def _phase(name: str) -> RefinementPhase:
    return RefinementPhase.make(Path(f"/fake/cifs/{name}.cif"))


def _make_result(rwp: float, rpb_value: float | None = None, peak_rows=None) -> RefinementResult:
    """Build a minimal but genuinely valid (pydantic-validated) RefinementResult.

    Only ``rwp``/``rpb`` and (for the integration test) ``peak_data`` are
    exercised by the code under test; everything else is a structurally
    valid placeholder.
    """
    if rpb_value is None:
        rpb_value = rwp
    return RefinementResult(
        lst_data=LstResult(
            raw_lst="",
            pattern_name="fake",
            num_steps=1,
            rp=0.0,
            rpb=rpb_value,
            r=0.0,
            rwp=rwp,
            rexp=1.0,
            d=0.0,
            rho=rpb_value,
            phases_results={},
        ),
        plot_data=DiaResult(x=[0.0], y_obs=[0.0], y_calc=[0.0], y_bkg=[0.0], structs={}),
        peak_data=pd.DataFrame(
            peak_rows or [{"phase": "", "2theta": 0.0, "intensity": 0.0}]
        ),
        refinement_metrics=RefinementMetrics(rwp=rwp),
    )


class TestKeepBranchDespiteIntensityOrder(unittest.TestCase):
    """Direct unit tests for the pure helper (Change 2)."""

    def test_not_violated_always_kept(self):
        parent = _make_result(rwp=50.0)
        child = _make_result(rwp=49.9)
        self.assertTrue(keep_branch_despite_intensity_order(parent, child, violated=False))

    def test_violated_with_material_improvement_kept(self):
        # 50 -> 40 is a 20% relative improvement, above the 8% threshold.
        parent = _make_result(rwp=50.0)
        child = _make_result(rwp=40.0)
        self.assertTrue(keep_branch_despite_intensity_order(parent, child, violated=True))

    def test_violated_with_marginal_improvement_pruned(self):
        # 50 -> 48 is a 4% relative improvement, below the 8% threshold.
        parent = _make_result(rwp=50.0)
        child = _make_result(rwp=48.0)
        self.assertFalse(keep_branch_despite_intensity_order(parent, child, violated=True))

    def test_improvement_exactly_at_threshold_is_kept(self):
        parent_rwp = 50.0
        child_rwp = parent_rwp * (1 - MATERIAL_RWP_IMPROVEMENT)
        parent = _make_result(rwp=parent_rwp)
        child = _make_result(rwp=child_rwp)
        self.assertTrue(keep_branch_despite_intensity_order(parent, child, violated=True))

    def test_violated_with_no_parent_falls_back_to_prune(self):
        child = _make_result(rwp=10.0)
        self.assertFalse(keep_branch_despite_intensity_order(None, child, violated=True))

    def test_violated_with_zero_parent_rwp_falls_back_to_prune(self):
        parent = _make_result(rwp=0.0)
        child = _make_result(rwp=0.0)
        self.assertFalse(keep_branch_despite_intensity_order(parent, child, violated=True))


class TestExpandNodeIntensityOrderIntegration(unittest.TestCase):
    """Exercise the real ``expand_node`` status decision (Change 2), not just the helper.

    ``score_phases``/``refine_phases`` (heavy, would otherwise run BGMN) and the
    unrelated overfitting check (``remove_unnecessary_phases``, which needs
    realistic plot data) are stubbed out. The intensity-order detection and the
    material-improvement gate run for real.
    """

    def setUp(self):
        self.phase_a = _phase("A")
        self.phase_b = _phase("B")

    def _build_tree(self, parent_result: RefinementResult) -> BaseSearchTree:
        tree = BaseSearchTree(
            pattern_path=Path("/fake/pattern.xy"),
            all_phases_result={self.phase_b: _make_result(rwp=1.0)},
            peak_obs=np.array([[10.0, 500.0], [20.0, 300.0]]),
            refine_params={},
            phase_params={},
            intensity_threshold=0.0,
            wavelength="Cu",
            instrument_profile="fake",
            express_mode=False,
            maximum_grouping_distance=0.1,
            max_phases=5,
            rpb_threshold=0.0,
            pinned_phases=[],
            record_peak_matcher_scores=False,
            peak_matching_strategy=None,
        )
        tree.add_node(
            Node(
                identifier="root",
                data=SearchNodeData(
                    current_result=parent_result, current_phases=[self.phase_a]
                ),
            )
        )
        return tree

    def _expand_with_child_rwp(self, parent_rwp: float, child_rwp: float) -> str:
        parent_result = _make_result(rwp=parent_rwp)
        # phase B (newly added) has more peak intensity than phase A (added
        # earlier) -> flags the intensity-order heuristic.
        child_result = _make_result(
            rwp=child_rwp,
            peak_rows=[
                {"phase": "A", "2theta": 10.0, "intensity": 50.0},
                {"phase": "B", "2theta": 20.0, "intensity": 500.0},
            ],
        )

        tree = self._build_tree(parent_result)
        tree.score_phases = lambda all_phases_result, current_result: ([self.phase_b], {}, 0.0)
        tree.refine_phases = lambda best_phases, pinned_phases: {self.phase_b: child_result}

        new_phases = [self.phase_a, self.phase_b]
        with (
            patch(
                "dara.search.tree.group_phases",
                return_value={self.phase_b: {"group_id": 0, "fom": 1.0, "lattice_strain": 0.0}},
            ),
            patch(
                "dara.search.tree.remove_unnecessary_phases",
                return_value=[p.path for p in new_phases],
            ),
        ):
            tree.expand_node("root")

        child_nid = next(nid for nid in tree.expand_tree("root") if nid != "root")
        return tree.get_node(child_nid).data.status

    def test_intensity_order_violation_with_material_improvement_is_kept(self):
        # 50 -> 30 is a 40% relative improvement: well above the threshold.
        status = self._expand_with_child_rwp(parent_rwp=50.0, child_rwp=30.0)
        self.assertEqual(status, "pending")

    def test_intensity_order_violation_below_threshold_is_pruned(self):
        # 50 -> 48 is a 4% relative improvement: below the threshold.
        status = self._expand_with_child_rwp(parent_rwp=50.0, child_rwp=48.0)
        self.assertEqual(status, "low_weight_fraction")


if __name__ == "__main__":
    unittest.main()
