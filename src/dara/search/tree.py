from __future__ import annotations

import copy
import warnings
from itertools import zip_longest
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Literal, TYPE_CHECKING

import numpy as np
import ray
from sklearn.cluster import AgglomerativeClustering
from treelib import Node, Tree

from dara import do_refinement_no_saving
from dara.cif2str import CIF2StrError
from dara.eflech_worker import EflechWorker
from dara.search.node import SearchNodeData
from dara.search.peak_matcher import PeakMatcher
from dara.utils import (
    get_number,
    get_optimal_max_two_theta,
    rpb,
    load_symmetrized_structure,
    DEPRECATED,
    get_logger,
    find_optimal_score_threshold,
    get_composition_distance,
    get_composition_from_filename,
)

if TYPE_CHECKING:
    from dara.result import RefinementResult


logger = get_logger(__name__, level="INFO")


@ray.remote(num_cpus=0)
class ExploredPhasesSet:
    """
    This is a remote actor to keep track of the explored phases in the search tree.

    This is used to avoid exploring the same phase combinations multiple times.
    """

    def __init__(self):
        self._set: set[frozenset[Path]] = set()

    def update(self, phases: list[tuple[Path, ...]]):
        for phase in phases:
            self._set.add(frozenset(phase))

    def multiple_contains(self, phases: list[tuple[Path, ...]]) -> list[bool]:
        return [frozenset(phases) in self._set for phases in phases]

    def contains_and_update(self, phases: list[tuple[Path, ...]]) -> list[bool]:
        """
        Check if the set contains the phases and update the set. This is an atomic operation.

        Args:
            phases: the phases to check

        Returns:
            a list of boolean values indicating whether the set contains the phases
        """
        contains = self.multiple_contains(phases)
        self.update(phases)
        return contains

    def get(self) -> frozenset[frozenset[Path]]:
        """
        Get the set of explored phases.

        Returns:
            the set of explored phases
        """
        return frozenset(self._set)


@ray.remote(num_cpus=1)
def remote_do_refinement_no_saving(
    pattern_path: Path,
    cif_paths: list[Path],
    instrument_name: str,
    phase_params: dict[str, ...] | None,
    refinement_params: dict[str, float] | None,
) -> RefinementResult | None:
    """
    Perform the actual refinement in the remote process.

    If the refinement fails, None will be returned.
    """
    if len(cif_paths) == 0:
        return None
    try:
        result = do_refinement_no_saving(
            pattern_path,
            cif_paths,
            instrument_name=instrument_name,
            phase_params=phase_params,
            refinement_params=refinement_params,
        )
    except (RuntimeError, TimeoutExpired, CIF2StrError) as e:
        logger.debug(f"Refinement failed for {cif_paths}, the reason is {e}")
        return None
    if result.lst_data.rpb == 100:
        logger.debug(f"Refinement failed for {cif_paths}, the reason is RPB = 100.")
        return None
    return result


@ray.remote(num_cpus=1)
def remote_peak_matching(
    batch: list[tuple[np.ndarray, np.ndarray]],
    return_type: Literal["PeakMatcher", "score", "jaccard"],
) -> list[PeakMatcher | float]:
    results = []

    for peak_calc, peak_obs in batch:
        pm = PeakMatcher(peak_calc, peak_obs)

        if return_type == "PeakMatcher":
            results.append(pm)
        elif return_type == "score":
            results.append(pm.score())
        elif return_type == "jaccard":
            results.append(pm.jaccard_index())
        else:
            raise ValueError(f"Unknown return type {return_type}")

    return results


def batch_peak_matching(
    peak_calcs: list[np.ndarray],
    peak_obs: np.ndarray | list[np.ndarray],
    return_type: Literal["PeakMatcher", "score", "jaccard"] = "PeakMatcher",
    batch_size: int = 100,
) -> list[PeakMatcher | float]:
    if isinstance(peak_obs, np.ndarray):
        peak_obs = [peak_obs] * len(peak_calcs)

    if len(peak_calcs) != len(peak_obs):
        raise ValueError("Length of peak_calcs and peak_obs must be the same.")

    all_data = list(zip_longest(peak_calcs, peak_obs, fillvalue=None))
    batches = [
        all_data[i : i + batch_size] for i in range(0, len(all_data), batch_size)
    ]
    handles = [
        remote_peak_matching.remote(batch, return_type=return_type) for batch in batches
    ]
    results = sum(ray.get(handles), [])

    return results


def batch_refinement(
    pattern_path: Path,
    cif_paths: list[list[Path]],
    instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    phase_params: dict[str, ...] | None = None,
    refinement_params: dict[str, float] | None = None,
) -> list[RefinementResult]:
    handles = [
        remote_do_refinement_no_saving.remote(
            pattern_path,
            cif_paths,
            instrument_name=instrument_name,
            phase_params=phase_params,
            refinement_params=refinement_params,
        )
        for cif_paths in cif_paths
    ]
    results = ray.get(handles)
    return results


def calculate_fom(phase_path: Path, result: RefinementResult) -> float:
    """
    Calculate the figure of merit for a phase.

    For more detail, refer to https://journals.iucr.org/j/issues/2019/03/00/nb5231/.
    Args:
        phase_path: the path to the phase (cif file)
        result: the refinement result

    Returns:
        the figure of merit of the target phase. If it cannot be calculated, return 0.
    """
    a = b = c = 1.0
    b1_threshold = 2e-2

    structure, _ = load_symmetrized_structure(phase_path)
    initial_lattice_abc = structure.lattice.abc

    refined_a = result.lst_data.phases_results[phase_path.stem].a
    refined_b = result.lst_data.phases_results[phase_path.stem].b
    refined_c = result.lst_data.phases_results[phase_path.stem].c

    geweicht = result.lst_data.phases_results[phase_path.stem].gewicht
    geweicht = get_number(geweicht)

    if hasattr(result.lst_data.phases_results[phase_path.stem], "B1"):
        b1 = get_number(result.lst_data.phases_results[phase_path.stem].B1) or 0
    else:
        b1 = 0

    if refined_a is None or geweicht is None:
        return 0

    refined_lattice_abc = [
        refined_a,
        refined_b if refined_b is not None else refined_a,
        refined_c if refined_c is not None else refined_a,
    ]
    refined_lattice_abc = [get_number(x) for x in refined_lattice_abc]

    initial_lattice_abc = np.array(initial_lattice_abc) / 10  # convert to nm
    refined_lattice_abc = np.array(refined_lattice_abc)

    delta_u = (
        np.sum(np.abs(initial_lattice_abc - refined_lattice_abc) / initial_lattice_abc)
        * 100
    )

    if delta_u <= 1:
        a = 0

    if b1 is None or b1 < b1_threshold:
        c = 0
    else:
        c /= b1

    return (1 / (result.lst_data.rho + a * delta_u + 1e-4) + b * geweicht) / (1 + c)


def group_phases(
    all_phases_result: dict[Path, RefinementResult | None],
    distance_threshold: float = 0.1,
) -> dict[Path, dict[str, float | int]]:
    """
    Group the phases based on their similarity.

    The similarity includes both the peak matching and the compositional similarity.

    Args:
        all_phases_result: the result of all the phases
        distance_threshold: the distance threshold for clustering, default to 0.1

    Returns:
        a dictionary containing the group id and the figure of merit for each phase
    """
    grouped_result = {}

    # handle the case where there is no result for a phase
    for phase, result in all_phases_result.items():
        if result is None:
            grouped_result[phase] = {"group_id": -1, "fom": 0}

    all_phases_result = {
        phase: result
        for phase, result in all_phases_result.items()
        if result is not None
    }

    if len(all_phases_result) <= 1:
        return {
            phase: {"group_id": 0, "fom": calculate_fom(phase, result)}
            for phase, result in all_phases_result.items()
        }

    peaks = []
    compositions = []

    for phase, result in all_phases_result.items():
        all_peaks = result.peak_data
        peaks.append(
            all_peaks[all_peaks["phase"] == phase.stem][["2theta", "intensity"]].values
        )
        compositions.append(get_composition_from_filename(phase))

    pairwise_similarity = batch_peak_matching(
        [p for p in peaks for _ in peaks],
        [p for _ in peaks for p in peaks],
        return_type="jaccard",
    )
    peal_distance_matrix = 1 - np.array(pairwise_similarity).reshape(
        len(peaks), len(peaks)
    )

    # current peak matching algorithm is not a symmetric metric.
    peal_distance_matrix = (peal_distance_matrix + peal_distance_matrix.T) / 2

    # calculate compositional distance matrix
    composition_distance_matrix = np.array(
        [
            [
                get_composition_distance(compositions[i], compositions[j])
                for j in range(len(compositions))
            ]
            for i in range(len(compositions))
        ]
    )

    # use the maximum distance
    distance_matrix = np.maximum(peal_distance_matrix, composition_distance_matrix)

    # clustering
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    clusterer.fit(distance_matrix)

    for i, cluster in enumerate(clusterer.labels_):
        phase = list(all_phases_result.keys())[i]
        result = list(all_phases_result.values())[i]
        grouped_result[phase] = {
            "group_id": cluster,
            "fom": calculate_fom(phase, result),
        }

    return grouped_result


def remove_unnecessary_phases(
    result: RefinementResult, cif_paths: list[Path], rpb_threshold: float = 1
) -> list[Path]:
    """
    Remove unnecessary phases from the result.

    If a phase cannot cause increase in RWP, it will be removed.
    """
    phases_results = {k: np.array(v) for k, v in result.plot_data.structs.items()}
    y_obs = np.array(result.plot_data.y_obs)
    y_calc = np.array(result.plot_data.y_calc)
    y_bkg = np.array(result.plot_data.y_bkg)

    cif_paths_dict = {cif_path.stem: cif_path for cif_path in cif_paths}

    original_rpb = rpb(y_calc, y_obs, y_bkg)

    new_phases = []

    for excluded_phase in phases_results:
        y_calc_excl = y_calc.copy()
        y_calc_excl -= phases_results[excluded_phase]

        new_rpb = rpb(y_calc_excl, y_obs, y_bkg)

        if new_rpb > original_rpb + rpb_threshold:
            new_phases.append(cif_paths_dict[excluded_phase])

    return new_phases


class BaseSearchTree(Tree):
    """
    A base class for the search tree. It is not intended to be used directly.

    Args:
        pattern_path: the path to the pattern
        all_phases_result: the result of all the phases
        peak_obs: the observed peaks
        rpb_threshold: the minimum RPB improvement in each step
        refine_params: the refinement parameters, it will be passed to the refinement function.
        phase_params: the phase parameters, it will be passed to the refinement function.
        instrument_name: the name of the instrument, it will be passed to the refinement function.
        maximum_grouping_distance: the maximum grouping distance, default to 0.1
        max_phases: the maximum number of phases
    """

    def __init__(
        self,
        pattern_path: Path,
        all_phases_result: dict[Path, RefinementResult] | None,
        peak_obs: np.ndarray | None,
        rpb_threshold: float,
        refine_params: dict[str, ...] | None,
        phase_params: dict[str, ...] | None,
        instrument_name: str,
        maximum_grouping_distance: float,
        max_phases: float,
        top_n: int = DEPRECATED,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # TODO: Remove top_n in the future
        if top_n != DEPRECATED:
            warnings.warn(
                "The top_n parameter is deprecated and will be removed in the future. "
                "You can set the peak_matcher_score_threshold to control the number of phases searched.",
            )

        self.pattern_path = pattern_path
        self.rpb_threshold = rpb_threshold
        self.refinement_params = refine_params if refine_params is not None else {}
        self.phase_params = phase_params if phase_params is not None else {}
        self.instrument_name = instrument_name
        self.maximum_grouping_distance = maximum_grouping_distance
        self.max_phases = max_phases

        self.all_phases_result = all_phases_result
        self.peak_obs = peak_obs

    def expand_node(
        self, nid: str, explored_phases_set: ExploredPhasesSet | None = None
    ) -> list[str]:
        """
        Expand a node in the search tree.

        This method will first do a naive search match method to find the best matched phases. Then it will refine the
        best matched phases and add the results to the search tree.

        Args:
            nid: the node id
            explored_phases_set: the set of explored phases. If it is not None, it will be used to avoid exploring the
                 same phase combinations multiple times.
        """
        node: Node = self.get_node(nid)
        logger.info(
            f"Expanding node {nid} with current phases {node.data.current_phases}, "
            f"Rwp = {node.data.current_result.lst_data.rwp if node.data.current_result is not None else None}"
        )
        if node is None:
            raise ValueError(f"Node with id {nid} does not exist.")
        if node.data is None or node.data.status != "pending":
            raise ValueError(f"Node with id {nid} is not expandable.")

        node.data.status = "running"
        try:
            # remove phases that are already in the current result
            current_phases_set = set(node.data.current_phases)
            all_phases_result = {
                phase: result
                for phase, result in self.all_phases_result.items()
                if phase not in current_phases_set
            }
            best_phases, scores, threshold = self.get_best_matched_phases(
                all_phases_result, node.data.current_result
            )

            if explored_phases_set is not None:
                explored = ray.get(
                    explored_phases_set.contains_and_update.remote(
                        [node.data.current_phases + [phase] for phase in best_phases]
                    )
                )
            else:
                explored = [False] * len(best_phases)

            for i in range(len(best_phases)):
                if explored[i]:
                    self.create_node(
                        data=SearchNodeData(
                            current_result=None,
                            current_phases=node.data.current_phases + [best_phases[i]],
                            status="duplicate",
                        ),
                        parent=nid,
                    )

            best_phases = [
                best_phases[i] for i in range(len(best_phases)) if not explored[i]
            ]

            node.data.peak_matcher_scores = scores
            node.data.peak_matcher_score_threshold = threshold

            new_results = self.get_all_phases_result(
                best_phases, pinned_phases=node.data.current_phases
            )

            # group the results
            grouped_results = group_phases(
                new_results,
                distance_threshold=self.maximum_grouping_distance,
            )

            for phase, new_result in new_results.items():
                new_phases = node.data.current_phases + [phase]

                group_id = grouped_results[phase]["group_id"]
                fom = grouped_results[phase]["fom"]
                is_best_result_in_group = fom == max(
                    [
                        grouped_results[phase_]["fom"]
                        for phase_ in grouped_results
                        if grouped_results[phase_]["group_id"] == group_id
                    ]
                )

                weight_fractions = (
                    new_result.get_phase_weights(normalize=True)
                    if new_result is not None
                    else None
                )

                if new_result is None:
                    status = "error"
                elif any(wt < 0.01 for wt in weight_fractions.values()):
                    status = "low_weight_fraction"
                elif node.data.current_result is not None and (
                    (
                        len(
                            remove_unnecessary_phases(
                                new_result, new_phases, rpb_threshold=self.rpb_threshold
                            )
                        )
                        != len(new_phases)
                    )
                    or (
                        new_result.lst_data.rpb
                        >= node.data.current_result.lst_data.rpb + self.rpb_threshold
                    )
                ):
                    status = "no_improvement"
                elif not is_best_result_in_group:
                    status = "similar_structure"
                elif len(new_phases) >= self.max_phases:
                    status = "max_depth"
                else:
                    status = "pending"

                self.create_node(
                    data=SearchNodeData(
                        current_result=new_result,
                        current_phases=new_phases,
                        status=status,
                        group_id=group_id,
                        fom=fom,
                    ),
                    parent=nid,
                )
        except Exception:
            node.data.status = "error"
            raise

        node.data.status = "expanded"

        return self.get_expandable_children(nid)

    def get_expandable_children(self, nid: str) -> list[str]:
        """
        Get the expandable children of a node.

        The expandable children are the children that have not been expanded yet, which is marked as "pending".

        Args:
            nid: the node id

        Returns:
            a list of node ids that are expandable
        """
        if not self.contains(nid):
            raise ValueError(f"Node with id {nid} does not exist.")

        return [
            child.identifier
            for child in self.children(nid)
            if self.get_node(child.identifier).data.status == "pending"
        ]

    def expand_root(
        self, explored_phases_set: ExploredPhasesSet | None = None
    ) -> list[str]:
        """
        Expand the root node.

        Args:
            explored_phases_set: the set of explored phases. If it is not None, it will be used to avoid exploring the
                    same phase combinations multiple times.
        """
        return self.expand_node(self.root, explored_phases_set=explored_phases_set)

    def get_search_results(self) -> dict[tuple[Path, ...], RefinementResult]:
        """
        Get the search results.

        The search results are the results of the nodes that have been expanded and have no expandable children.

        Returns:
            a dictionary containing the phase combinations and their results
        """
        results = {}
        all_phases = {}
        for nid, node in self.nodes.items():
            all_phases.setdefault(frozenset(node.data.current_phases), []).append(nid)

        for node in self.nodes.values():
            if node.data.status in {"expanded", "max_depth"} and all(
                child.data.status not in {"expanded", "max_depth"}
                for child in self.children(node.identifier)
            ):
                other_phases = all_phases[frozenset(node.data.current_phases)]
                if any(
                    self.get_node(nid).data.status
                    not in {"expanded", "max_depth", "duplicate"}
                    for nid in other_phases
                ):
                    continue
                results[tuple(node.data.current_phases)] = node.data.current_result
        return results

    def get_best_matched_phases(
        self,
        all_phases_result: dict[Path, RefinementResult],
        current_result: RefinementResult | None = None,
    ) -> tuple[list[Path], dict[Path, float], float]:
        """
        Get the best matched phases.

        This is a naive search-match method based on the peak matching score. It will return the best matched phases,
        all phases' scores, and the score's threshold.

        The threshold is determined by finding the inflection point of the percentile of the scores.

        Args:
            all_phases_result: the result of all the phases
            current_result: the current result

        Returns:
            a tuple containing the best matched phases, all phases' scores, and the score's threshold
        """
        if current_result is None:
            missing_peaks = self.peak_obs
        else:
            current_peak_calc = current_result.peak_data[["2theta", "intensity"]].values
            missing_peaks = PeakMatcher(current_peak_calc, self.peak_obs).missing

        if len(missing_peaks) == 0:
            return [], {}, 0

        peak_calcs = [
            refinement_result.peak_data[
                refinement_result.peak_data["phase"] == phase.stem
            ][["2theta", "intensity"]].values
            for phase, refinement_result in all_phases_result.items()
        ]
        scores = dict(
            zip_longest(
                all_phases_result.keys(),
                batch_peak_matching(peak_calcs, missing_peaks, return_type="score"),
                fillvalue=None,
            )
        )

        peak_matcher_score_threshold, _ = find_optimal_score_threshold(
            list(scores.values())
        )
        peak_matcher_score_threshold = max(peak_matcher_score_threshold, 0)

        filtered_scores = {
            phase: score
            for phase, score in scores.items()
            if score >= peak_matcher_score_threshold
        }

        return (
            sorted(filtered_scores, key=lambda x: filtered_scores[x], reverse=True),
            scores,
            peak_matcher_score_threshold,
        )

    def get_all_phases_result(
        self, phases: list[Path], pinned_phases: list[Path] | None = None
    ) -> dict[Path, RefinementResult | None]:
        """
        Get the result of all the phases.

        Args:
            phases: the phases
            pinned_phases: the pinned phases thta will be included in all the refinement

        Returns:
            a dictionary containing the phase and its result
        """
        if pinned_phases is None:
            pinned_phases = []

        all_phases_result = dict(
            zip_longest(
                phases,
                self._batch_refine([[phase] + pinned_phases for phase in phases]),
                fillvalue=None,
            )
        )
        return all_phases_result

    def _batch_refine(
        self,
        all_references: list[list[Path]],
    ) -> list[RefinementResult]:
        return batch_refinement(
            self.pattern_path,
            all_references,
            instrument_name=self.instrument_name,
            phase_params=self.phase_params,
            refinement_params=self.refinement_params,
        )

    def _clone(self, identifier=None, with_tree=False, deep=False):
        return self.__class__(
            identifier=identifier,
            tree=self if with_tree else None,
            deep=deep,
            max_phases=self.max_phases,
            pattern_path=self.pattern_path,
            all_phases_result=self.all_phases_result,
            peak_obs=self.peak_obs,
            rpb_threshold=self.rpb_threshold,
            refine_params=self.refinement_params,
            phase_params=self.phase_params,
            instrument_name=self.instrument_name,
            maximum_grouping_distance=self.maximum_grouping_distance,
        )

    @classmethod
    def from_search_tree(
        cls, root_nid: str, search_tree: BaseSearchTree
    ) -> BaseSearchTree:
        """
        Create a new search tree from an existing search tree.

        Args:
            root_nid: the node id that will be used as the root node for the new search tree
            search_tree: the search tree that will be used to create the new search tree

        Returns:
            the new search tree
        """
        root_node = search_tree.get_node(root_nid)
        if root_node is None:
            raise ValueError(f"Node with id {root_nid} does not exist.")

        new_search_tree = cls(
            max_phases=search_tree.max_phases,
            pattern_path=search_tree.pattern_path,
            all_phases_result=search_tree.all_phases_result,
            peak_obs=search_tree.peak_obs,
            rpb_threshold=search_tree.rpb_threshold,
            refine_params=search_tree.refinement_params,
            phase_params=search_tree.phase_params,
            instrument_name=search_tree.instrument_name,
            maximum_grouping_distance=search_tree.maximum_grouping_distance,
        )
        new_search_tree.add_node(root_node)

        return new_search_tree

    def add_subtree(self, anchor_nid: str, search_tree: BaseSearchTree):
        """
        Add a subtree to the search tree.

        Args:
            anchor_nid: the node id that the subtree will be added to
            search_tree: the search tree that will be added to the search tree

        Returns:
            the merged search tree
        """
        # update the data from the search tree
        if (
            search_tree.get_node(search_tree.root).data.current_phases
            != self.get_node(anchor_nid).data.current_phases
        ):
            raise ValueError(
                "The root node of the subtree must have the same current_phases as the anchor node."
            )

        self.merge(nid=anchor_nid, new_tree=search_tree, deep=False)
        self.update_node(anchor_nid, data=search_tree.get_node(search_tree.root).data)


class SearchTree(BaseSearchTree):
    """
    A class for the search tree.

    Args:
        pattern_path: the path to the pattern
        cif_paths: the paths to the CIF files
        pinned_phases: the phases that will be included in all the refinement
        rpb_threshold: the minimum RPB improvement in each step
        refine_params: the refinement parameters, it will be passed to the refinement function.
        phase_params: the phase parameters, it will be passed to the refinement function.
        instrument_name: the name of the instrument, it will be passed to the refinement function.
        maximum_grouping_distance: the maximum grouping distance, default to 0.1
        max_phases: the maximum number of phases, note that the pinned phases are COUNTED as well
    """

    def __init__(
        self,
        pattern_path: Path,
        cif_paths: list[Path],
        pinned_phases: list[Path] | None = None,
        rpb_threshold: float = 2,
        refine_params: dict[str, ...] | None = None,
        phase_params: dict[str, ...] | None = None,
        instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
        maximum_grouping_distance: float = 0.1,
        max_phases: float = 5,
        *args,
        **kwargs,
    ):
        self.pinned_phases = pinned_phases if pinned_phases is not None else []
        self.cif_paths = cif_paths

        if len(self.pinned_phases) >= max_phases:
            raise ValueError(
                "The number of pinned phases must be less than the max_phases, "
                "as the pinned phases are counted in the max_phases."
            )

        super().__init__(
            pattern_path=pattern_path,
            all_phases_result=None,  # placeholder, will be updated later
            peak_obs=None,  # placeholder, will be updated later
            rpb_threshold=rpb_threshold,
            refine_params=refine_params,
            phase_params=phase_params,
            instrument_name=instrument_name,
            maximum_grouping_distance=maximum_grouping_distance,
            max_phases=max_phases,
            *args,
            **kwargs,
        )
        root_node = self._create_root_node()
        self.add_node(root_node)

        peak_obs = self._detect_peak_in_pattern()
        self.peak_obs = peak_obs

        all_phases_result = self._get_all_cleaned_phases_result()
        self.all_phases_result = all_phases_result

    def _detect_peak_in_pattern(self) -> np.ndarray:
        logger.info("Detecting peaks in the pattern.")
        if self.refinement_params.get("wmax", None) is not None:
            warnings.warn(
                f"The wmax ({self.refinement_params['wmax']}) in refinement_params "
                f"will be ignored. The wmax will be automatically adjusted."
            )
        eflech_worker = EflechWorker()
        peak_list = eflech_worker.run_peak_detection(
            self.pattern_path, wmin=self.refinement_params.get("wmin", None), wmax=None
        )
        optimal_wmax = get_optimal_max_two_theta(peak_list)
        logger.info(f"The wmax is automatically adjusted to {optimal_wmax}.")
        self.refinement_params["wmax"] = optimal_wmax

        peak_list_array = peak_list[["2theta", "intensity"]].values

        return peak_list_array[
            np.where(peak_list_array[:, 0] < self.refinement_params["wmax"])
        ]

    def _create_root_node(self) -> Node:
        logger.info("Creating the root node.")
        root_node = Node(
            data=SearchNodeData(
                current_result=(
                    self._batch_refine([self.pinned_phases])[0]
                    if self.pinned_phases
                    else None
                ),
                current_phases=self.pinned_phases,
            ),
        )
        return root_node

    def _get_all_cleaned_phases_result(self) -> dict[Path, RefinementResult]:
        logger.info("Refining all the phases in the dataset.")
        pinned_phases_set = set(self.pinned_phases)
        cif_paths = [
            cif_path for cif_path in self.cif_paths if cif_path not in pinned_phases_set
        ]
        all_phases_result = self.get_all_phases_result(
            cif_paths, pinned_phases=self.pinned_phases
        )

        # clean up cif paths (if no result, remove from list)
        all_phases_result = {
            phase: result
            for phase, result in all_phases_result.items()
            if result is not None
        }

        logger.info(
            f"Finished refining {len(all_phases_result)} phases, "
            f"with {len(cif_paths) - len(all_phases_result)} phases removed."
        )

        return all_phases_result

    def _clone(self, identifier=None, with_tree=False, deep=False):
        raise NotImplementedError(f"{self.__class__.__name__} cannot be cloned.")
