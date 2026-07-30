import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from treelib import Node, Tree

from dara.refine import RefinementPhase
from dara.result import DiaResult, LstResult, PhaseResult, RefinementResult
from dara.search.data_model import SearchNodeData
from dara.search.tree import (
    LOW_WEIGHT_FRACTION_RWP_IMPROVEMENT,
    BaseSearchTree,
    should_prune_low_weight_fraction,
)


def _phase(name: str) -> RefinementPhase:
    return RefinementPhase.make(Path(f"/fake/cifs/{name}.cif"))


def _phase_result() -> PhaseResult:
    """A structurally-valid but otherwise-meaningless PhaseResult.

    `calculate_fom_and_strain` (the only code that reads these fields) is
    mocked out in the integration test below, so content doesn't matter --
    it only needs to satisfy LstResult.phases_results' pydantic validation.
    """
    return PhaseResult(
        SpacegroupNo=None,
        HermannMauguin=None,
        XrayDensity=None,
        Rphase=None,
        UNIT="nm",
        GEWICHT=1.0,
        GEWICHT_NAME=None,
    )


def _make_refinement_result(
    rwp: float,
    phase_intensities: dict[str, float],
) -> RefinementResult:
    """Build a minimal but genuinely valid (pydantic-validated) RefinementResult.

    `phase_intensities` maps phase stem -> total calculated peak intensity
    for that phase; both `peak_data` (read by the intensity-ordering check)
    and `plot_data.structs` (read by `remove_unnecessary_phases`) are built
    from it so removing any one phase changes the calculated pattern enough
    for `remove_unnecessary_phases` to consider it necessary.
    """
    peak_rows = [
        {"phase": phase, "2theta": 10.0 + i, "intensity": intensity}
        for i, (phase, intensity) in enumerate(phase_intensities.items())
    ]
    structs = {phase: [intensity] for phase, intensity in phase_intensities.items()}
    y_calc_total = sum(phase_intensities.values())

    return RefinementResult(
        lst_data=LstResult(
            raw_lst="",
            pattern_name="fake",
            num_steps=1,
            Rp=0.0,
            Rpb=rwp,
            R=0.0,
            Rwp=rwp,
            Rexp=1.0,
            d=0.0,
            **{"1-rho": rwp},
            phases_results={phase: _phase_result() for phase in phase_intensities},
        ),
        plot_data=DiaResult(
            x=[0.0],
            y_obs=[y_calc_total],
            y_calc=[y_calc_total],
            y_bkg=[0.0],
            structs=structs,
        ),
        peak_data=pd.DataFrame(peak_rows),
    )


class TestShouldPruneLowWeightFraction(unittest.TestCase):
    """Direct unit tests for the standalone pruning-decision helper."""

    def test_in_order_never_pruned(self):
        """Not out-of-order: the parent/child fit is irrelevant, never prune."""
        self.assertFalse(
            should_prune_low_weight_fraction(intensity_out_of_order=False, parent_rwp=40.0, child_rwp=39.9)
        )
        self.assertFalse(
            should_prune_low_weight_fraction(intensity_out_of_order=False, parent_rwp=40.0, child_rwp=100.0)
        )
        self.assertFalse(
            should_prune_low_weight_fraction(intensity_out_of_order=False, parent_rwp=None, child_rwp=39.9)
        )

    def test_out_of_order_no_parent_falls_back_to_ordering_only(self):
        """With no parent to compare against (root's first child), the
        ordering signal alone determines the outcome -- prune."""
        self.assertTrue(should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=None, child_rwp=10.0))

    def test_out_of_order_zero_or_negative_parent_rwp_is_pruned(self):
        """Degenerate parent_rwp <= 0 can't support a relative-improvement
        calculation (division by zero); fall back to the ordering-only rule."""
        self.assertTrue(should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=0.0, child_rwp=5.0))
        self.assertTrue(should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=-1.0, child_rwp=5.0))

    def test_out_of_order_marginal_improvement_still_pruned(self):
        """Out-of-order + improvement below the material threshold: pruned."""
        parent_rwp = 40.0
        # 5% relative improvement, below the 10% default threshold
        child_rwp = 38.0
        self.assertTrue(
            should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=parent_rwp, child_rwp=child_rwp)
        )

    def test_out_of_order_material_improvement_kept(self):
        """Out-of-order + material improvement: this is the bug fix -- a
        branch that fits clearly better than its parent must survive even
        though the newest phase isn't the smallest."""
        parent_rwp = 40.0
        # 50% relative improvement
        child_rwp = 20.0
        self.assertFalse(
            should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=parent_rwp, child_rwp=child_rwp)
        )

    def test_improvement_exactly_at_threshold_is_kept(self):
        """Exactly LOW_WEIGHT_FRACTION_RWP_IMPROVEMENT relative improvement
        counts as material (the check is `< threshold`, not `<= threshold`)."""
        parent_rwp = 40.0
        child_rwp = parent_rwp * (1 - LOW_WEIGHT_FRACTION_RWP_IMPROVEMENT)
        self.assertFalse(
            should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=parent_rwp, child_rwp=child_rwp)
        )

    def test_improvement_just_under_threshold_is_pruned(self):
        parent_rwp = 40.0
        child_rwp = parent_rwp * (1 - LOW_WEIGHT_FRACTION_RWP_IMPROVEMENT) + 1e-6
        self.assertTrue(
            should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=parent_rwp, child_rwp=child_rwp)
        )

    def test_worse_fit_is_pruned(self):
        """Out-of-order + the child fits WORSE than the parent (negative
        'improvement'): definitely pruned."""
        self.assertTrue(should_prune_low_weight_fraction(intensity_out_of_order=True, parent_rwp=20.0, child_rwp=30.0))

    def test_custom_threshold_is_respected(self):
        parent_rwp, child_rwp = 40.0, 30.0  # 25% relative improvement
        self.assertTrue(
            should_prune_low_weight_fraction(
                intensity_out_of_order=True,
                parent_rwp=parent_rwp,
                child_rwp=child_rwp,
                material_improvement_threshold=0.5,
            )
        )
        self.assertFalse(
            should_prune_low_weight_fraction(
                intensity_out_of_order=True,
                parent_rwp=parent_rwp,
                child_rwp=child_rwp,
                material_improvement_threshold=0.2,
            )
        )


def _make_test_tree() -> BaseSearchTree:
    """A minimal, real (treelib-backed) BaseSearchTree for driving the actual
    `expand_node()` code path without needing a real pattern file or BGMN.
    """
    tree = BaseSearchTree.__new__(BaseSearchTree)
    Tree.__init__(tree)
    tree.pinned_phases = []
    tree.all_phases_result = {}
    tree.max_phases = 5
    tree.rpb_threshold = 0.5
    tree.record_peak_matcher_scores = False
    tree.express_mode = False
    tree.maximum_grouping_distance = 0.1
    tree.peak_obs = pd.DataFrame({"2theta": [10.0, 11.0], "intensity": [100.0, 50.0]})[["2theta", "intensity"]].values
    return tree


@patch("dara.search.tree.calculate_fom_and_strain", return_value=(0.5, 0.0))
class TestExpandNodeLowWeightFractionIntegration(unittest.TestCase):
    """Integration-level tests exercising the real `expand_node()` method
    (real `remove_unnecessary_phases`, real intensity-ordering computation,
    real status-decision branching) with `score_phases`/`refine_phases`
    mocked to avoid needing BGMN, and `calculate_fom_and_strain` mocked
    since it reads a phase's own CIF file (irrelevant to this decision).
    """

    def _run_expand(self, parent_rwp, child_rwp, phase_a_intensity, phase_b_intensity):
        """Set up a root -> phase_a chain, then expand with phase_b added.

        phase_b's calculated intensity is set higher than phase_a's, so the
        addition is out-of-order by construction; parent_rwp/child_rwp
        control whether it's a material improvement.
        """
        tree = _make_test_tree()
        phase_a, phase_b = _phase("A"), _phase("B")

        parent_result = _make_refinement_result(parent_rwp, {"A": phase_a_intensity})
        root = Node(
            data=SearchNodeData(
                current_result=parent_result,
                current_phases=[phase_a],
                status="pending",
            )
        )
        tree.add_node(root)

        child_result = _make_refinement_result(child_rwp, {"A": phase_a_intensity, "B": phase_b_intensity})

        with (
            patch.object(tree, "score_phases", return_value=([phase_b], {}, 0)),
            patch.object(tree, "refine_phases", return_value={phase_b: child_result}),
        ):
            tree.expand_node(root.identifier)

        children = tree.children(root.identifier)
        self.assertEqual(len(children), 1)
        return children[0].data.status

    def test_material_improvement_is_retained(self, mock_fom):
        """Out-of-order (B's intensity > A's) but a clearly better fit
        (50% relative Rwp improvement): must NOT be pruned as low-weight-fraction."""
        status = self._run_expand(parent_rwp=40.0, child_rwp=20.0, phase_a_intensity=60.0, phase_b_intensity=200.0)
        self.assertNotEqual(status, "low_weight_fraction")
        self.assertEqual(status, "pending")

    def test_marginal_improvement_is_still_pruned(self, mock_fom):
        """Out-of-order (B's intensity > A's) with only a marginal (5%)
        Rwp improvement: still pruned as low-weight-fraction, as before."""
        status = self._run_expand(parent_rwp=40.0, child_rwp=38.0, phase_a_intensity=60.0, phase_b_intensity=200.0)
        self.assertEqual(status, "low_weight_fraction")

    def test_in_order_addition_never_flagged_regardless_of_fit(self, mock_fom):
        """B's intensity is LOWER than A's (properly ordered): never flagged,
        even with a poor Rwp improvement -- the ordering check simply
        doesn't fire, independent of the material-improvement carve-out."""
        status = self._run_expand(parent_rwp=40.0, child_rwp=39.9, phase_a_intensity=200.0, phase_b_intensity=60.0)
        self.assertNotEqual(status, "low_weight_fraction")


if __name__ == "__main__":
    unittest.main()
