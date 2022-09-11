"""Neuron tree."""

from typing import Any, Callable, TypeVar, cast, overload

import matplotlib.axes
import matplotlib.collections
import numpy as np
import numpy.typing as npt

from ..utils import painter, padding1d
from ._node import NodeAttached

__all__ = ["Tree"]

T, K = TypeVar("T"), TypeVar("K")


class Tree:
    """A neuron tree, which should be a binary tree in most cases."""

    class Node(NodeAttached):
        """Node of neuron tree."""

        def __init__(self, tree: "Tree", idx: int) -> None:
            super().__init__(tree, idx)

    ndata: dict[str, npt.NDArray[Any]]
    source: str | None

    def __init__(
        self,
        *,
        typee: npt.NDArray[np.int32] | None = None,
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        pid: npt.NDArray[np.int32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        n_nodes = self.number_of_nodes()
        pid = pid or np.arange(-1, n_nodes - 1, step=1, dtype=np.int32)
        self.ndata = {
            "id": np.arange(0, n_nodes, step=1, dtype=np.int32),
            "type": padding1d(n_nodes, typee, dtype=np.int32),
            "x": padding1d(n_nodes, x),
            "y": padding1d(n_nodes, y),
            "z": padding1d(n_nodes, z),
            "r": padding1d(n_nodes, r, padding_value=1),
            "pid": padding1d(n_nodes, pid, dtype=np.int32),
            **kwargs,
        }

        self.source = None

    def __len__(self) -> int:
        """Get number of nodes."""
        return self.number_of_nodes()

    def __repr__(self) -> str:
        nodes, edges = self.number_of_nodes(), self.number_of_edges()
        return f"Neuron Tree with {nodes} nodes and {edges} edges"

    def __getitem__(self, idx: int) -> Node:
        """Get node by id."""
        return self.Node(self, idx)

    # fmt:off
    def id(self)   -> npt.NDArray[np.int32]:   return self.ndata["id"]  # pylint: disable=invalid-name
    def type(self) -> npt.NDArray[np.int32]:   return self.ndata["type"]
    def x(self)    -> npt.NDArray[np.float32]: return self.ndata["x"]
    def y(self)    -> npt.NDArray[np.float32]: return self.ndata["y"]
    def z(self)    -> npt.NDArray[np.float32]: return self.ndata["z"]
    def r(self)    -> npt.NDArray[np.float32]: return self.ndata["r"]
    def pid(self)  -> npt.NDArray[np.int32]:   return self.ndata["pid"]
    # fmt:on

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get array of shape(N, 3)."""
        return np.array([self.x(), self.y(), self.z()])

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get array of shape(N, 4)."""
        return np.array([self.x(), self.y(), self.z(), self.r()])

    def number_of_nodes(self) -> int:
        """Get the number of nodes."""
        return self.id().shape[0]

    def number_of_edges(self) -> int:
        """Get the number of edges."""
        return self.number_of_nodes() - 1

    def to_swc(self, swc_path: str) -> None:
        """Write swc file."""
        ids = self.id()
        types = self.type()
        xyzr = self.xyzr()
        pid = self.pid()

        def get_line_str(idx: int) -> str:
            x, y, z, r = [f"{f:.4f}" for f in xyzr[idx]]
            items = [ids[idx], types[idx], x, y, z, r, pid[idx]]
            return " ".join(map(str, items))

        with open(swc_path, "w", encoding="utf-8") as f:
            f.write(f"# source: {self.source if self.source else 'Unknown'}\n")
            f.write("# id type x y z r pid\n")
            f.writelines(map(get_line_str, ids))

    def draw(
        self,
        color: str | None = painter.palette.momo,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.collections.LineCollection]:
        """Draw neuron tree.

        Parameters
        ----------
        color : str, optional
            Color of branch. If `None`, the default color will be enabled.
        ax : ~matplotlib.axes.Axes, optional
            A subplot of `~matplotlib`. If `None`, a new one will be created.
        **kwargs : dict[str, Unknown]
            Forwarded to `~matplotlib.collections.LineCollection`.

        Returns
        -------
        ax : ~matplotlib.axes.Axes
            If provided, return as-is.
        collection : ~matplotlib.collections.LineCollection
            Drawn line collection.
        """
        xyz = self.xyz()  # (N, 3)
        edges = np.array([xyz[range(self.number_of_nodes())], xyz[self.pid()]])
        return painter.draw_lines(edges, ax=ax, color=color, **kwargs)

    TraverseEnter = Callable[[Node, T | None], T]
    TraverseLeave = Callable[[Node, list[T]], T]

    # fmt:off
    @overload
    def traverse(self, *, enter: TraverseEnter[T]) -> None: ...
    @overload
    def traverse(self, *, enter: TraverseEnter[T] | None = None, leave: TraverseLeave[K]) -> K: ...
    # fmt:on

    def traverse(
        self,
        *,
        enter: TraverseEnter[T] | None = None,
        leave: TraverseLeave[K] | None = None,
    ) -> K | None:
        """Traverse each nodes.

        Parameters
        ----------
        enter : Callable[[Node, list[T]], T], optional
            The callback when entering each node, it accepts two parameters,
            the first parameter is the current node, the second parameter is
            the parent's information T, and the root node receives an None.
        leave : Callable[[Node, T | None], T], optional
            The callback when leaving each node. When leaving a node, subtree
            has already been traversed. Callback accepts two parameters, the
            first parameter is the current node, the second parameter is the
            children's information T, and the leaf node receives an empty list.
        """

        children_map = dict[int, list[int]]()
        for pid in self.pid():
            if pid == -1:
                continue

            children_map.setdefault(pid, [])
            children_map[pid].append(pid)

        def dfs(
            idx: int,
            enter: Tree.TraverseEnter[T] | None,
            leave: Tree.TraverseLeave[K] | None,
            pre: T | None,
        ) -> K | None:
            cur = enter(self[idx], pre) if enter is not None else None
            children = [dfs(i, enter, leave, cur) for i in children_map.get(idx, [])]
            children = cast(list[K], children)
            return leave(self[idx], children) if leave is not None else None

        return dfs(0, enter, leave, None)

    @classmethod
    def copy(cls) -> "Tree":
        """Make a copy."""

        new_tree = cls(**{k: v.copy() for k, v in cls.ndata.items()})
        new_tree.source = cls.source
        return new_tree

    @classmethod
    def normalize(cls) -> "Tree":
        """Scale the `x`, `y`, `z`, `r` of nodes to 0-1."""
        new_tree = cls.copy()
        for key in ["x", "y", "z", "r"]:  # TODO: does r is the same?
            v_max = np.max(new_tree.ndata[key])
            v_min = np.min(new_tree.ndata[key])
            new_tree.ndata[key] = (new_tree.ndata[key] - v_min) / v_max

        return new_tree
