"""Phase search module."""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import jenkspy
import ray

from dara.search.tree import SearchTree, BaseSearchTree, ExploredPhasesSet
from dara.utils import DEPRECATED, get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from dara.result import RefinementResult


logger = get_logger(__name__)


def remove_duplicate_results(
    results: dict[tuple[Path, ...], RefinementResult],
) -> dict[tuple[Path, ...], RefinementResult]:
    """
    Remove duplicate results.

    If two results have the same phases, only the one with the lower RWP will be kept.
    """
    results_ = {}
    appeared_phases = set()

    for phases, result in results.items():
        if set(phases) not in appeared_phases:
            results_[phases] = result
            appeared_phases.add(frozenset(phases))

    return results_


@ray.remote
def _remote_expand_node(
    search_tree: BaseSearchTree, explored_phases_set: ExploredPhasesSet
) -> BaseSearchTree:
    """Expand a node in the search tree."""
    search_tree.expand_root(explored_phases_set=explored_phases_set)
    return search_tree


def remote_expand_node(
    search_tree: SearchTree, nid: str, explored_phases_set: ExploredPhasesSet
) -> ray.ObjectRef:
    """Expand a node in the search tree."""
    subtree = BaseSearchTree.from_search_tree(root_nid=nid, search_tree=search_tree)
    return _remote_expand_node.remote(subtree, explored_phases_set)


def search_phases(
    pattern_path: Path,
    cif_paths: list[Path],
    pinned_phases: list[Path] | None = None,
    max_phases: int = 5,
    rpb_threshold: float = 2,
    return_search_tree: bool = False,
    top_n: int = DEPRECATED,
) -> dict[tuple[Path, ...], RefinementResult] | SearchTree:
    """Search for the best phases to use for refinement."""
    phase_params = {
        "gewicht": "0_0",
        "lattice_range": 0.01,
        "k1": "0_0^0.01",
        "k2": "fixed",
        "b1": "0_0^0.005",
        "rp": 4,
    }
    refinement_params = {"n_threads": 8}

    # TODO: remove top_n in the future
    # build the search tree
    search_tree = SearchTree(
        pattern_path=pattern_path,
        cif_paths=cif_paths,
        pinned_phases=pinned_phases,
        rpb_threshold=rpb_threshold,
        refine_params=refinement_params,
        phase_params=phase_params,
        max_phases=max_phases,
        top_n=top_n,
    )

    max_worker = ray.cluster_resources()["CPU"]
    explored_phases_set = ExploredPhasesSet.remote()
    pending = [remote_expand_node(search_tree, search_tree.root, explored_phases_set)]
    to_be_submitted = deque()

    while pending:
        done, pending = ray.wait(pending, timeout=0.5)

        for task in done:
            remote_search_tree = ray.get(task)
            search_tree.add_subtree(
                anchor_nid=remote_search_tree.root, search_tree=remote_search_tree
            )
            for nid in search_tree.get_expandable_children(remote_search_tree.root):
                to_be_submitted.append(nid)

        while len(pending) < max_worker and to_be_submitted:
            nid = to_be_submitted.popleft()
            pending.append(remote_expand_node(search_tree, nid, explored_phases_set))

    if not return_search_tree:
        results = search_tree.get_search_results()
        all_rhos = [result.lst_data.rho for result in results.values()]

        # TODO: use a better method to remove bad results
        if len(all_rhos) > 5:
            # get the first natural break
            interval = jenkspy.jenks_breaks(all_rhos, n_classes=2)
            rho_cutoff = interval[1]
            results = {k: v for k, v in results.items() if v.lst_data.rho <= rho_cutoff}

        results = remove_duplicate_results(results)
        return results
    else:
        return search_tree
