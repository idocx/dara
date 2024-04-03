"""Phase search module."""
from __future__ import annotations

import copy
from collections import deque
from traceback import print_exc
from typing import TYPE_CHECKING

import ray

from dara.search.data_model import SearchResult
from dara.search.tree import BaseSearchTree, SearchTree
from dara.utils import DEPRECATED

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_PHASE_PARAMS = {
    "gewicht": "0_0",
    "lattice_range": 0.01,
    "k1": "0_0^0.01",
    "k2": "fixed",
    "b1": "0_0^0.005",
    "rp": 4,
}
DEFAULT_REFINEMENT_PARAMS = {"n_threads": 8}


@ray.remote
def _remote_expand_node(search_tree: BaseSearchTree) -> BaseSearchTree:
    """Expand a node in the search tree."""
    try:
        search_tree.expand_root()
        return search_tree
    except Exception as e:
        print_exc()
        raise e


def remote_expand_node(search_tree: SearchTree, nid: str) -> ray.ObjectRef:
    """Expand a node in the search tree."""
    subtree = BaseSearchTree.from_search_tree(root_nid=nid, search_tree=search_tree)
    return _remote_expand_node.remote(subtree)


def search_phases(
    pattern_path: Path | str,
    cif_paths: list[Path | str],
    pinned_phases: list[Path | str] | None = None,
    max_phases: int = 5,
    instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    phase_params: dict[str, ...] | None = None,
    refinement_params: dict[str, ...] | None = None,
    return_search_tree: bool = False,
    rpb_threshold: float = DEPRECATED,
) -> list[SearchResult] | SearchTree:
    """
    Search for the best phases to use for refinement.

    Args:
        pattern_path: the path to the pattern file. It has to be either in `.xrdml` or `.xy` format
        cif_paths: the paths to the CIF files
        pinned_phases: the paths to the pinned phases, which will be included in all the results
        max_phases: the maximum number of phases to refine
        instrument_name: the name of the instrument
        phase_params: the parameters for the phase search
        refinement_params: the parameters for the refinement
        return_search_tree: whether to return the search tree. This is mainly used for debugging purposes.
        rpb_threshold: the RPB threshold, which is deprecated, and will be removed in the future
    """
    if phase_params is None:
        phase_params = {}

    if refinement_params is None:
        refinement_params = {}

    phase_params = {**DEFAULT_PHASE_PARAMS, **phase_params}
    refinement_params = {**DEFAULT_REFINEMENT_PARAMS, **refinement_params}

    # build the search tree
    search_tree = SearchTree(
        pattern_path=pattern_path,
        cif_paths=cif_paths,
        pinned_phases=pinned_phases,
        rpb_threshold=rpb_threshold,
        instrument_name=instrument_name,
        refine_params=refinement_params,
        phase_params=phase_params,
        max_phases=max_phases,
    )

    max_worker = ray.cluster_resources()["CPU"]
    pending = [remote_expand_node(search_tree, search_tree.root)]
    to_be_submitted = deque()

    while pending:
        done, pending = ray.wait(pending, timeout=0.5)

        for task in done:
            remote_search_tree = ray.get(task)
            remote_search_tree = copy.deepcopy(remote_search_tree)
            search_tree.add_subtree(
                anchor_nid=remote_search_tree.root, search_tree=remote_search_tree
            )
            for nid in search_tree.get_expandable_children(remote_search_tree.root):
                to_be_submitted.append(nid)

        while len(pending) < max_worker and to_be_submitted:
            nid = to_be_submitted.popleft()
            pending.append(remote_expand_node(search_tree, nid))

    if not return_search_tree:
        return search_tree.get_search_results()
    else:
        return search_tree
