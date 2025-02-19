"""Transformation in tree."""

from typing import Callable, List, Tuple

import numpy as np

from ..core import BranchTree, Tree, cut_tree, to_subtree
from .base import Transform

__all__ = [
    "ToBranchTree",
    "TreeNormalizer",
    "CutByBifurcationOrder",
    "CutShortTipBranch",
]


# pylint: disable=too-few-public-methods
class ToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    def __call__(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)


# pylint: disable=too-few-public-methods
class TreeNormalizer(Transform[Tree, Tree]):
    """Noramlize coordinates and radius to 0-1."""

    def __call__(self, x: Tree) -> "Tree":
        """Scale the `x`, `y`, `z`, `r` of nodes to 0-1."""
        new_tree = x.copy()
        for key in ["x", "y", "z", "r"]:  # TODO: does r is the same?
            vs = new_tree.ndata[key]
            new_tree.ndata[key] = (vs - np.min(vs)) / np.max(vs)

        return new_tree


class CutByBifurcationOrder(Transform[Tree, Tree]):
    """Cut tree by bifurcation order."""

    max_bifurcation_order: int

    def __init__(self, max_bifurcation_order: int) -> None:
        self.max_bifurcation_order = max_bifurcation_order

    def __call__(self, x: Tree) -> Tree:
        return cut_tree(x, enter=self._enter)

    def __repr__(self) -> str:
        return f"CutByBifurcationOrder-{self.max_bifurcation_order}"

    def _enter(self, n: Tree.Node, parent_level: int | None) -> Tuple[int, bool]:
        if parent_level is None:
            level = 0
        elif n.is_bifurcation():
            level = parent_level + 1
        else:
            level = parent_level
        return (level, level >= self.max_bifurcation_order)


class CutShortTipBranch(Transform[Tree, Tree]):
    """Cut off too short terminal branches.

    This method is usually applied in the post-processing of manual
    reconstruction. When the user draw lines, a line head is often left
    at the junction of two lines.
    """

    thre: float
    callback: Callable[[Tree.Branch], None] | None

    def __init__(
        self, thre: float = 5, callback: Callable[[Tree.Branch], None] | None = None
    ) -> None:
        self.thre = thre
        self.callback = callback

    def __repr__(self) -> str:
        return f"CutShortTipBranch-{self.thre}"

    def __call__(self, x: Tree) -> Tree:
        removals: List[int] = []

        def collect_short_branch(
            n: Tree.Node, children: List[Tuple[float, Tree.Node] | None]
        ) -> Tuple[float, Tree.Node] | None:
            if len(children) == 0:  # tip
                return 0, n

            if len(children) == 1:
                if children[0] is None:
                    return None

                dis, child = children[0]
                dis += n.distance(child)
                return dis, n

            for c in children:
                if c is None:
                    continue

                dis, child = c
                if dis + n.distance(child) <= self.thre:
                    removals.append(child.id)  # TODO: change this to a callback

                    if self.callback is not None:
                        path = [n.id]  # n does not delete, but will include in callback
                        while child is not None:  # TODO: perf
                            path.append(child.id)
                            child = cc[0] if len((cc := child.children())) > 0 else None
                        self.callback(Tree.Branch(n.attach, path))

            return None

        x.traverse(leave=collect_short_branch)
        return to_subtree(x, removals)
