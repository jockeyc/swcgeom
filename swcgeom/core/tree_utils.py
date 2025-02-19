"""SWC util wrapper for tree."""

import warnings
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar, overload

import numpy as np

from .swc import SWCLike
from .swc_utils import (
    REMOVAL,
    Topology,
    propagate_removal,
    sort_nodes_impl,
    to_sub_topology,
    traverse,
)
from .tree import Tree

__all__ = ["sort_tree", "cut_tree", "to_sub_tree", "to_subtree", "get_subtree"]

T, K = TypeVar("T"), TypeVar("K")


def sort_tree(tree: Tree) -> Tree:
    """Sort the indices of neuron tree.

    See Also
    --------
    ~swc_utils.sort_nodes
    """
    (new_ids, new_pids), id_map = sort_nodes_impl((tree.id(), tree.pid()))
    new_tree = tree.copy()
    new_tree.ndata = {k: tree.ndata[k][id_map] for k in tree.ndata}
    new_tree.ndata.update(id=new_ids, pid=new_pids)
    return new_tree


# fmt:off
@overload
def cut_tree(tree: Tree, *, enter: Callable[[Tree.Node, T | None], Tuple[T, bool]]) -> Tree: ...
@overload
def cut_tree(tree: Tree, *, leave: Callable[[Tree.Node, List[K]], Tuple[K, bool]]) -> Tree: ...
# fmt:on


def cut_tree(tree: Tree, *, enter=None, leave=None):
    """Traverse and cut the tree.

    Returning a `True` can delete the current node and its children.
    """

    removals: List[int] = []

    if enter:

        def _enter(n: Tree.Node, parent: Tuple[T, bool] | None) -> Tuple[T, bool]:
            if parent is not None and parent[1]:
                removals.append(n.id)
                return parent

            res, removal = enter(n, parent[0] if parent else None)
            if removal:
                removals.append(n.id)

            return res, removal

        tree.traverse(enter=_enter)

    elif leave:

        def _leave(n: Tree.Node, children: List[K]) -> K:
            res, removal = leave(n, children)
            if removal:
                removals.append(n.id)

            return res

        tree.traverse(leave=_leave)

    else:
        return tree.copy()

    return to_subtree(tree, removals)


def to_sub_tree(swc_like: SWCLike, sub: Topology) -> Tuple[Tree, Dict[int, int]]:
    """Create subtree from origin tree.

    .. deprecated:: 0.6.0
        `to_sub_tree` will be removed in v0.6.0, it is replaced by
        `to_subtree` beacuse it is easy to use.

    You can directly mark the node for removal, and we will remove it,
    but if the node you remove is not a leaf node, you need to use
    `propagate_remove` to remove all children.

    Returns
    -------
    tree : Tree
    id_map : Dict[int, int]
    """
    warnings.warn(
        "`to_sub_tree` will be removed in v0.6.0, it is replaced by "
        "`to_subtree` beacuse it is easy to use."
    )

    sub = propagate_removal(sub)
    (new_id, new_pid), id_map_arr = to_sub_topology(sub)

    n_nodes = new_id.shape[0]
    ndata = {k: swc_like.get_ndata(k)[id_map_arr].copy() for k in swc_like.keys()}
    ndata.update(id=new_id, pid=new_pid)

    subtree = Tree(n_nodes, **ndata)
    subtree.source = swc_like.source

    id_map = {}
    for i, idx in enumerate(id_map_arr):
        id_map[idx] = i
    return subtree, id_map


def to_subtree(swc_like: SWCLike, removals: Iterable[int]) -> Tree:
    """Create subtree from origin tree.

    Parameters
    ----------
    swc_like : SWCLike
    removals : List of int
        A list of id of nodes to be removed.
    """
    new_ids = swc_like.id().copy()
    for i in removals:
        new_ids[i] = REMOVAL

    sub = propagate_removal((new_ids, swc_like.pid()))
    return _to_subtree(swc_like, sub)


def get_subtree(swc_like: SWCLike, n: int) -> Tree:
    """Get subtree rooted at n.

    Parameters
    ----------
    swc_like : SWCLike
    n : int
        Id of the root of the subtree.
    """
    ids = []
    topo = (swc_like.id(), swc_like.pid())
    traverse(topo, enter=lambda n, _: ids.append(n), root=n)

    sub_ids = np.array(ids, dtype=np.int32)
    sub = (sub_ids, swc_like.pid()[sub_ids])
    return _to_subtree(swc_like, sub)


def _to_subtree(swc_like: SWCLike, sub: Topology) -> Tree:
    (new_id, new_pid), id_map = to_sub_topology(sub)

    n_nodes = new_id.shape[0]
    ndata = {k: swc_like.get_ndata(k)[id_map].copy() for k in swc_like.keys()}
    ndata.update(id=new_id, pid=new_pid)

    subtree = Tree(n_nodes, **ndata)
    subtree.source = swc_like.source
    return subtree
