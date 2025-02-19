"""Branch tree is a simplified neuron tree."""

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

from .branch import Branch
from .swc_utils import to_sub_topology
from .tree import Tree

__all__ = ["BranchTree"]


class BranchTree(Tree):
    """Branch tree keeps topology of tree.

    A branch tree that contains only soma, branch, and tip nodes.
    """

    branches: Dict[int, List[Branch]]

    def get_origin_branches(self) -> list[Branch]:
        """Get branches of original tree."""
        return list(itertools.chain(*self.branches.values()))

    def get_origin_node_branches(self, idx: int) -> List[Branch]:
        """Get branches of node of original tree."""
        return self.branches[idx]

    @staticmethod
    def from_tree(tree: Tree) -> "BranchTree":
        """Generating a branch tree from tree."""

        branches = tree.get_branches()

        sub_id = np.array([0] + [br[-1].id for br in branches], dtype=np.int32)
        sub_pid = np.array([-1] + [br[0].id for br in branches], dtype=np.int32)

        (new_id, new_pid), id_map = to_sub_topology((sub_id, sub_pid))

        n_nodes = new_id.shape[0]
        ndata = {k: tree.get_ndata(k)[id_map].copy() for k in tree.keys()}
        ndata.update(id=new_id, pid=new_pid)

        branch_tree = BranchTree(n_nodes, **ndata)
        branch_tree.source = tree.source  # TODO

        branch_tree.branches = {}
        for branch_raw in branches:
            idx = np.nonzero(id_map == branch_raw[0].id)[0].item()
            branch_tree.branches.setdefault(idx, [])
            branch_tree.branches[idx].append(branch_raw.detach())

        return branch_tree

    @classmethod
    def from_data_frame(cls, df: pd.DataFrame, *args, **kwargs) -> "BranchTree":
        tree = super().from_data_frame(df, *args, **kwargs)
        return BranchTree.from_tree(tree)
