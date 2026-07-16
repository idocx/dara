import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from treelib import Node, Tree

from dara.refine import RefinementPhase
from dara.result import DiaResult, LstResult, RefinementMetrics, RefinementResult
from dara.search.data_model import SearchNodeData
from dara.search.tree import (
    LOW_WEIGHT_FRACTION_MATERIAL_RWP_IMPROVEMENT,
    BaseSearchTree,
    attempt_minimal_phase_recovery,
    check_low_weight_fraction,
    find_minimal_phase_set,
    is_recovery_material,
)


def _make_result(intensities: dict[str, float], rwp: float) -> SimpleNamespace:
    """Build a minimal duck-typed stand-in for a RefinementResult.

    ``check_low_weight_fraction`` only reads ``result.peak_data`` (a DataFrame
    with "phase"/"2theta"/"intensity" columns) and ``result.lst_data.rwp``, so
    a full pydantic ``RefinementResult`` (which also requires valid
    ``plot_data``/``refinement_metrics``) is unnecessary for this unit test.
    """
    rows = [
        {"phase": phase, "2theta": 10.0, "intensity": intensity}
        for phase, intensity in intensities.items()
    ]
    return SimpleNamespace(
        peak_data=pd.DataFrame(rows),
        lst_data=SimpleNamespace(rwp=rwp),
    )


def _phase(name: str) -> RefinementPhase:
    return RefinementPhase.make(Path(f"/fake/cifs/{name}.cif"))


class TestCheckLowWeightFraction(unittest.TestCase):
    """Test the fit-aware low-weight-fraction pruning criterion.

    Real numbers in some of these cases are taken directly from the RPA1_7
    regression investigation (CLUSTERING_ANALYSIS.md, "Why 141 loses to 61"):
    at scan 30 (700C), `SrCO3 -> TiO2_61_(icsd_145381) -> TiO2_141` refined to
    rwp 22.99 from a parent (`SrCO3 -> TiO2_61_(icsd_145381)` alone) of rwp
    47.31 -- a ~51% relative improvement -- while the newly-added TiO2_141
    (intensity ~162) was larger than the previously-added TiO2_61_icsd_145381
    (intensity ~60). The old ordering-only rule discarded this branch; the fix
    should keep it.
    """

    def setUp(self):
        self.srco3 = _phase("SrCO3_62")
        self.tio2_61 = _phase("TiO2_61_icsd_145381")
        self.tio2_141 = _phase("TiO2_141")

    def test_single_searched_phase_never_flagged(self):
        """At the root, only one phase is ever searched -- trivially in order."""
        result = _make_result({"SrCO3_62": 900.0}, rwp=48.66)
        flagged = check_low_weight_fraction(
            new_phases=[self.srco3],
            pinned_phases=[],
            new_result=result,
            parent_result=None,
        )
        self.assertFalse(flagged)

    def test_properly_ordered_addition_not_flagged(self):
        """Newest phase has less intensity than the earlier one: never suspicious."""
        result = _make_result(
            {"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0}, rwp=47.31
        )
        flagged = check_low_weight_fraction(
            new_phases=[self.srco3, self.tio2_61],
            pinned_phases=[],
            new_result=result,
            parent_result=_make_result({"SrCO3_62": 900.0}, rwp=48.66),
        )
        self.assertFalse(flagged)

    def test_out_of_order_with_marginal_improvement_still_flagged(self):
        """Out-of-order + improvement below the material threshold: still pruned.

        Mirrors the real SrCO3+TiO2_141+TiO2_61(cod) branch at scan 30
        (47.05 -> 43.91, ~6.7% relative improvement): the pruning intent
        (catch marginal, likely-overfitting additions) should be preserved.
        """
        parent = _make_result({"SrCO3_62": 900.0, "TiO2_141": 157.0}, rwp=47.05)
        result = _make_result(
            {"SrCO3_62": 900.0, "TiO2_141": 157.0, "TiO2_61_icsd_145381": 200.0},
            rwp=43.91,
        )
        flagged = check_low_weight_fraction(
            new_phases=[self.srco3, self.tio2_141, self.tio2_61],
            pinned_phases=[],
            new_result=result,
            parent_result=parent,
        )
        self.assertTrue(flagged)

    def test_out_of_order_with_material_improvement_not_flagged(self):
        """Out-of-order + material improvement: this is the bug fix.

        Real numbers from RPA1_7 scan 30: TiO2_141 (intensity ~162) is added
        after TiO2_61_icsd_145381 (intensity ~60), violating the strict
        decreasing-intensity rule, but the fit improves from rwp 47.31 to
        22.99 (~51% relative) -- real signal, not overfitting.
        """
        parent = _make_result({"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0}, rwp=47.31)
        result = _make_result(
            {"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0, "TiO2_141": 162.0},
            rwp=22.99,
        )
        flagged = check_low_weight_fraction(
            new_phases=[self.srco3, self.tio2_61, self.tio2_141],
            pinned_phases=[],
            new_result=result,
            parent_result=parent,
        )
        self.assertFalse(flagged)

    def test_improvement_exactly_at_threshold_boundary(self):
        """Improvement exactly at the configured threshold counts as material."""
        parent_rwp = 40.0
        new_rwp = parent_rwp * (1 - LOW_WEIGHT_FRACTION_MATERIAL_RWP_IMPROVEMENT)
        parent = _make_result({"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0}, rwp=parent_rwp)
        result = _make_result(
            {"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0, "TiO2_141": 162.0},
            rwp=new_rwp,
        )
        flagged = check_low_weight_fraction(
            new_phases=[self.srco3, self.tio2_61, self.tio2_141],
            pinned_phases=[],
            new_result=result,
            parent_result=parent,
        )
        self.assertFalse(flagged)

    def test_out_of_order_with_no_parent_result_still_flagged(self):
        """With no parent to compare against, fall back to the ordering-only rule."""
        result = _make_result(
            {"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0, "TiO2_141": 162.0},
            rwp=22.99,
        )
        flagged = check_low_weight_fraction(
            new_phases=[self.srco3, self.tio2_61, self.tio2_141],
            pinned_phases=[],
            new_result=result,
            parent_result=None,
        )
        self.assertTrue(flagged)

    def test_pinned_phases_excluded_from_ordering_check(self):
        """A large pinned phase shouldn't count against the ordering of searched phases."""
        parent = _make_result(
            {"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0}, rwp=47.31
        )
        result = _make_result(
            {"SrCO3_62": 900.0, "TiO2_61_icsd_145381": 60.0, "TiO2_141": 30.0},
            rwp=45.0,
        )
        flagged = check_low_weight_fraction(
            new_phases=[self.srco3, self.tio2_61, self.tio2_141],
            pinned_phases=[self.srco3],  # SrCO3 (intensity 900) is pinned, excluded
            new_result=result,
            parent_result=parent,
        )
        # searched phases are just [TiO2_61_icsd_145381 (60), TiO2_141 (30)],
        # already in decreasing order -> not flagged regardless of SrCO3's size
        self.assertFalse(flagged)


def _make_refinement_result(rpb: float, rwp: float | None = None) -> RefinementResult:
    """Build a minimal but genuinely valid (pydantic-validated) RefinementResult.

    Unlike `check_low_weight_fraction`, the recovery machinery constructs
    real `SearchNodeData(current_result=...)` instances internally, which
    pydantic-validates the `current_result` field as an actual
    `RefinementResult` -- a duck-typed `SimpleNamespace` won't pass. `rpb`/
    `rwp` are the only fields these tests care about; everything else is a
    structurally-valid placeholder.
    """
    if rwp is None:
        rwp = rpb
    return RefinementResult(
        lst_data=LstResult(
            raw_lst="",
            pattern_name="fake",
            num_steps=1,
            rp=0.0,
            rpb=rpb,
            r=0.0,
            rwp=rwp,
            rexp=1.0,
            d=0.0,
            rho=rpb,
            phases_results={},
        ),
        plot_data=DiaResult(x=[0.0], y_obs=[0.0], y_calc=[0.0], y_bkg=[0.0], structs={}),
        peak_data=pd.DataFrame({"phase": [], "2theta": [], "intensity": []}),
        refinement_metrics=RefinementMetrics(rwp=rwp),
    )


class TestFindMinimalPhaseSet(unittest.TestCase):
    def test_drops_unnecessary_phase_preserving_order(self):
        a, b, c, d = (_phase(n) for n in ["A", "B", "C", "D"])
        result = find_minimal_phase_set(
            new_phases=[a, b, c, d],
            necessary_paths=[a.path, c.path, d.path],
        )
        self.assertEqual(result, [a, c, d])

    def test_nothing_dropped_when_all_necessary(self):
        a, b = (_phase(n) for n in ["A", "B"])
        result = find_minimal_phase_set(
            new_phases=[a, b], necessary_paths=[a.path, b.path]
        )
        self.assertEqual(result, [a, b])


class TestIsRecoveryMaterial(unittest.TestCase):
    def test_strictly_better_is_material(self):
        minimal = SimpleNamespace(lst_data=SimpleNamespace(rpb=20.0))
        original = SimpleNamespace(lst_data=SimpleNamespace(rpb=50.0))
        self.assertTrue(is_recovery_material(minimal, original, rpb_threshold=1.0))

    def test_within_threshold_is_material(self):
        minimal = SimpleNamespace(lst_data=SimpleNamespace(rpb=50.5))
        original = SimpleNamespace(lst_data=SimpleNamespace(rpb=50.0))
        self.assertTrue(is_recovery_material(minimal, original, rpb_threshold=1.0))

    def test_worse_beyond_threshold_is_not_material(self):
        minimal = SimpleNamespace(lst_data=SimpleNamespace(rpb=52.0))
        original = SimpleNamespace(lst_data=SimpleNamespace(rpb=50.0))
        self.assertFalse(is_recovery_material(minimal, original, rpb_threshold=1.0))


def _make_test_tree(pinned_phases=None, max_phases=5) -> BaseSearchTree:
    """Build a minimal, real (treelib-backed) BaseSearchTree for testing the
    recovery machinery's tree-mutation logic in isolation, without needing a
    real pattern file or BGMN. `_batch_refine` is left unset -- tests that
    need it (missing-intermediate-node cases) patch it directly.
    """
    tree = BaseSearchTree.__new__(BaseSearchTree)
    Tree.__init__(tree)
    tree.pinned_phases = pinned_phases or []
    tree.max_phases = max_phases
    tree.rpb_threshold = 1.0
    tree._recovery_group_id_counter = 10**6

    root_node = Node(
        data=SearchNodeData.model_construct(
            current_result=None,
            current_phases=list(tree.pinned_phases),
            status="expanded",
            group_id=-1,
            fom=0,
            lattice_strain=0,
        )
    )
    tree.add_node(root_node)
    return tree


def _add_child(tree: BaseSearchTree, parent_nid: str, phases: list, status: str, rpb: float):
    node = Node(
        data=SearchNodeData.model_construct(
            current_result=SimpleNamespace(lst_data=SimpleNamespace(rpb=rpb, rwp=rpb)),
            current_phases=phases,
            status=status,
            group_id=0,
            fom=0,
            lattice_strain=0,
        )
    )
    tree.add_node(node, parent=parent_nid)
    return node


@patch("dara.search.tree.calculate_fom_and_strain", return_value=(0.5, 0.0))
class TestRecoverMinimalPhaseBranch(unittest.TestCase):
    """`calculate_fom_and_strain` is patched at module level in every test
    here: it normally reads a phase's own CIF (`load_symmetrized_structure`)
    plus its refined lattice/weight from the result, neither of which is
    relevant to whether the tree-mutation logic (reuse vs. create, status,
    parenting) is correct.
    """

    def test_reuses_existing_node_without_new_refinement(self, _mock_fom):
        tree = _make_test_tree()
        y = _phase("Y")
        # "Y alone" already exists as some other branch's root child
        existing = _add_child(tree, tree.root, [y], status="expanded", rpb=30.0)

        minimal_result = _make_refinement_result(rpb=20.0)
        newly_pending = tree._recover_minimal_phase_branch([y], minimal_result)

        # reused, not newly created -> never reported, even though nothing failed
        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 2)  # root + the pre-existing node only
        # the existing node's own data must be untouched, not overwritten
        self.assertEqual(tree.get_node(existing.identifier).data.status, "expanded")

    def test_reused_existing_pending_node_not_reported_either(self, _mock_fom):
        """The bug this guards against: reusing an *already-pending* node
        must not report it again, or the caller would submit a second,
        duplicate `expand_node` ray task for the same node id.
        """
        tree = _make_test_tree()
        y = _phase("Y")
        existing = _add_child(tree, tree.root, [y], status="pending", rpb=30.0)

        minimal_result = _make_refinement_result(rpb=20.0)
        newly_pending = tree._recover_minimal_phase_branch([y], minimal_result)

        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 2)
        self.assertEqual(tree.get_node(existing.identifier).data.status, "pending")

    def test_creates_new_single_phase_node(self, _mock_fom):
        tree = _make_test_tree(max_phases=5)
        y = _phase("Y")
        minimal_result = _make_refinement_result(rpb=20.0)

        newly_pending = tree._recover_minimal_phase_branch([y], minimal_result)

        self.assertEqual(len(tree.nodes), 2)  # root + 1 new node
        self.assertEqual(len(newly_pending), 1)
        new_node = tree.get_node(newly_pending[0])
        self.assertEqual(new_node.data.current_phases, [y])
        self.assertEqual(new_node.data.status, "pending")
        self.assertIs(new_node.data.current_result, minimal_result)
        # group_id must not collide with a normal sibling-clustering group id
        self.assertGreaterEqual(new_node.data.group_id, 10**6)

    def test_creates_missing_intermediate_node(self, _mock_fom):
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        intermediate_result = _make_refinement_result(rpb=40.0)
        minimal_result = _make_refinement_result(rpb=20.0)
        tree._batch_refine = lambda all_references: [intermediate_result]

        newly_pending = tree._recover_minimal_phase_branch([x, y], minimal_result)

        self.assertEqual(len(tree.nodes), 3)  # root + "X" + "X, Y"
        # both the intermediate ("X") and final ("X, Y") nodes are newly
        # created and under max_phases, so both must be reported
        self.assertEqual(len(newly_pending), 2)
        intermediate_node = tree.get_node(newly_pending[0])
        final_node = tree.get_node(newly_pending[1])
        self.assertEqual(intermediate_node.data.current_phases, [x])
        self.assertIs(intermediate_node.data.current_result, intermediate_result)
        self.assertEqual(intermediate_node.data.status, "pending")
        self.assertEqual(final_node.data.current_phases, [x, y])
        self.assertIs(final_node.data.current_result, minimal_result)
        self.assertEqual(tree.parent(final_node.identifier).identifier, intermediate_node.identifier)

    def test_max_depth_status_when_at_cap(self, _mock_fom):
        tree = _make_test_tree(max_phases=1)
        y = _phase("Y")
        minimal_result = _make_refinement_result(rpb=20.0)

        newly_pending = tree._recover_minimal_phase_branch([y], minimal_result)

        # created, but at the depth cap -> "max_depth", not "pending"
        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 2)
        created_node = next(n for n in tree.nodes.values() if n.data.current_phases == [y])
        self.assertEqual(created_node.data.status, "max_depth")

    def test_pinned_only_phases_do_nothing(self, _mock_fom):
        srco3 = _phase("SrCO3_62")
        tree = _make_test_tree(pinned_phases=[srco3])
        minimal_result = _make_refinement_result(rpb=20.0)

        newly_pending = tree._recover_minimal_phase_branch([srco3], minimal_result)

        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 1)  # just root, nothing created

    def test_aborts_cleanly_when_intermediate_refinement_fails(self, _mock_fom):
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        minimal_result = _make_refinement_result(rpb=20.0)
        tree._batch_refine = lambda all_references: [None]  # simulates a failed refinement

        newly_pending = tree._recover_minimal_phase_branch([x, y], minimal_result)

        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 1)  # nothing partially created

    def test_does_not_reuse_no_improvement_node_as_chain_attachment(self, _mock_fom):
        """Regression test for the real 'Node ... is not expanded' crash on
        scan 37: a phase-matching node with status "no_improvement" must
        never be reused as a chain attachment point, since nothing ever
        resubmits it for expansion -- it would permanently strand any child
        attached under it as an unexpandable ancestor.
        """
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        # "X" already exists, but as a normal terminal "no_improvement" node
        # -- e.g. it failed the rpb-improvement check against its own parent
        # earlier in the search. It must never gain a child.
        no_improvement_x = _add_child(tree, tree.root, [x], status="no_improvement", rpb=50.0)

        minimal_result = _make_refinement_result(rpb=20.0)
        tree._batch_refine = lambda all_references: [_make_refinement_result(rpb=45.0)]
        newly_pending = tree._recover_minimal_phase_branch([x, y], minimal_result)

        # the "no_improvement" node must remain untouched and childless
        self.assertEqual(tree.get_node(no_improvement_x.identifier).data.status, "no_improvement")
        self.assertEqual(len(tree.children(no_improvement_x.identifier)), 0)

        # a FRESH sibling "X" node was created instead, with its own proper
        # pending lifecycle, and the [X, Y] chain hangs off of *that*
        self.assertEqual(len(newly_pending), 2)  # fresh "X" and "X, Y"
        fresh_x = tree.get_node(newly_pending[0])
        self.assertEqual(fresh_x.data.current_phases, [x])
        self.assertEqual(fresh_x.data.status, "pending")
        self.assertNotEqual(fresh_x.identifier, no_improvement_x.identifier)

        final_node = tree.get_node(newly_pending[1])
        self.assertEqual(final_node.data.current_phases, [x, y])
        self.assertEqual(tree.parent(final_node.identifier).identifier, fresh_x.identifier)

        # the crash reproduction: walking ancestors from the final node must
        # not raise, since every ancestor (the fresh "X" node, once it
        # reaches "expanded") is now a valid, eventually-expandable node --
        # simulate it having been properly processed by the search loop.
        fresh_x.data.status = "expanded"
        final_node.data.status = "expanded"
        try:
            tree.get_phase_combinations(final_node)
        except ValueError:
            self.fail("get_phase_combinations raised -- the crash was not fixed")

    def test_does_not_reuse_low_weight_fraction_node_as_chain_attachment(self, _mock_fom):
        """Same regression, for the other normally-terminal status."""
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        low_weight_x = _add_child(tree, tree.root, [x], status="low_weight_fraction", rpb=50.0)

        minimal_result = _make_refinement_result(rpb=20.0)
        tree._batch_refine = lambda all_references: [_make_refinement_result(rpb=45.0)]
        newly_pending = tree._recover_minimal_phase_branch([x, y], minimal_result)

        self.assertEqual(len(tree.children(low_weight_x.identifier)), 0)
        self.assertEqual(len(newly_pending), 2)
        fresh_x = tree.get_node(newly_pending[0])
        self.assertNotEqual(fresh_x.identifier, low_weight_x.identifier)
        self.assertEqual(fresh_x.data.status, "pending")

    def test_still_reuses_pending_and_similar_structure_nodes(self, _mock_fom):
        """The fix must not be over-broad: "pending" and "similar_structure"
        remain reusable, since both either already satisfy or will
        eventually satisfy the ancestor-status check.
        """
        tree = _make_test_tree(max_phases=5)
        y = _phase("Y")
        for status in ("pending", "similar_structure", "expanded", "max_depth"):
            with self.subTest(status=status):
                existing = _add_child(tree, tree.root, [y], status=status, rpb=30.0)
                minimal_result = _make_refinement_result(rpb=20.0)
                newly_pending = tree._recover_minimal_phase_branch([y], minimal_result)
                self.assertEqual(newly_pending, [])  # reused, nothing created
                self.assertEqual(tree.get_node(existing.identifier).data.status, status)
                tree.remove_node(existing.identifier)  # reset for next subTest


@patch("dara.search.tree.calculate_fom_and_strain", return_value=(0.5, 0.0))
class TestAttemptMinimalPhaseRecovery(unittest.TestCase):
    """Integration-level tests of the orchestration function, exercising
    real `remove_unnecessary_phases` (not mocked -- it's cheap, pure numpy
    over `plot_data`) against real (but tiny) RefinementResult objects.
    """

    def _make_bloated_node(self, tree, parent_nid, phases, rpb, keep_indices):
        """Build a 'no_improvement' node whose plot_data is rigged so that
        `remove_unnecessary_phases` finds exactly `keep_indices` necessary.
        """
        structs = {}
        y_calc_total = 0.0
        for i, phase in enumerate(phases):
            contribution = 100.0 if i in keep_indices else 0.0
            structs[phase.path.stem] = [contribution]
            y_calc_total += contribution
        result = RefinementResult(
            lst_data=LstResult(
                raw_lst="",
                pattern_name="fake",
                num_steps=1,
                rp=0.0,
                rpb=rpb,
                r=0.0,
                rwp=rpb,
                rexp=1.0,
                d=0.0,
                rho=rpb,
                phases_results={},
            ),
            plot_data=DiaResult(
                x=[0.0],
                y_obs=[y_calc_total],
                y_calc=[y_calc_total],
                y_bkg=[0.0],
                structs=structs,
            ),
            peak_data=pd.DataFrame({"phase": [], "2theta": [], "intensity": []}),
            refinement_metrics=RefinementMetrics(rwp=rpb),
        )
        node = Node(
            data=SearchNodeData.model_construct(
                current_result=result,
                current_phases=phases,
                status="no_improvement",
                group_id=0,
                fom=0,
                lattice_strain=0,
            )
        )
        tree.add_node(node, parent=parent_nid)
        return node

    def test_skips_no_improvement_node_with_no_redundant_phase(self, _mock_fom):
        tree = _make_test_tree()
        x, y = _phase("X"), _phase("Y")
        # both phases contribute -> remove_unnecessary_phases keeps both
        self._make_bloated_node(tree, tree.root, [x, y], rpb=50.0, keep_indices={0, 1})

        newly_pending = attempt_minimal_phase_recovery(tree, tree.root)

        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 2)  # root + the untouched bloated node

    def test_recovers_and_reports_pending_node(self, _mock_fom):
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        # x contributes nothing -> removable; y is the real signal
        self._make_bloated_node(tree, tree.root, [x, y], rpb=50.0, keep_indices={1})
        tree._batch_refine = lambda all_references: [_make_refinement_result(rpb=22.0)]

        newly_pending = attempt_minimal_phase_recovery(tree, tree.root)

        self.assertEqual(len(newly_pending), 1)
        recovered = tree.get_node(newly_pending[0])
        self.assertEqual(recovered.data.current_phases, [y])
        self.assertEqual(recovered.data.status, "pending")

    def test_does_not_spawn_when_reduced_set_fits_worse(self, _mock_fom):
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        self._make_bloated_node(tree, tree.root, [x, y], rpb=50.0, keep_indices={1})
        # the reduced set actually fits much worse when really refined
        tree._batch_refine = lambda all_references: [_make_refinement_result(rpb=90.0)]

        newly_pending = attempt_minimal_phase_recovery(tree, tree.root)

        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 2)  # no recovery node added

    def test_reused_existing_expanded_node_not_reported_as_pending(self, _mock_fom):
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        self._make_bloated_node(tree, tree.root, [x, y], rpb=50.0, keep_indices={1})
        # "Y alone" already exists elsewhere in the tree, already resolved
        _add_child(tree, tree.root, [y], status="expanded", rpb=22.0)
        tree._batch_refine = lambda all_references: [_make_refinement_result(rpb=22.0)]

        newly_pending = attempt_minimal_phase_recovery(tree, tree.root)

        self.assertEqual(newly_pending, [])  # reused, not newly pending
        self.assertEqual(len(tree.nodes), 3)  # root + bloated + the pre-existing "Y"

    def test_reused_existing_pending_node_not_double_reported(self, _mock_fom):
        """Regression test for a real bug caught before landing: reusing an
        already-`"pending"` node (e.g. one created by normal search and not
        yet expanded) must not be reported again here, or the orchestrator
        would submit two concurrent `expand_node` ray tasks for the same
        node id -- which is exactly what happened in the first version of
        this fix (observed as an orphaned `"pending"` node left over at the
        end of a real 9-scan sweep on RPA1_7 scan 32).
        """
        tree = _make_test_tree(max_phases=5)
        x, y = _phase("X"), _phase("Y")
        self._make_bloated_node(tree, tree.root, [x, y], rpb=50.0, keep_indices={1})
        # "Y alone" already exists but hasn't been expanded by normal search yet
        _add_child(tree, tree.root, [y], status="pending", rpb=22.0)
        tree._batch_refine = lambda all_references: [_make_refinement_result(rpb=22.0)]

        newly_pending = attempt_minimal_phase_recovery(tree, tree.root)

        self.assertEqual(newly_pending, [])
        self.assertEqual(len(tree.nodes), 3)  # root + bloated + the pre-existing "Y"


if __name__ == "__main__":
    unittest.main()
