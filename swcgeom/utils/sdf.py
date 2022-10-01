"""Signed distance functions.

Refs: https://iquilezles.org/articles/distfunctions/
"""

from typing import List

import numpy as np
import numpy.typing as npt

__all__ = ["SDF", "SDFCompose", "SDFRoundCone"]


class SDF:
    """Signed distance functions."""

    def __call__(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self.distance(p)

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calc signed distance.

        Parmeters
        ---------
        p: ArrayLike
            Hit point p of shape (N, 3).

        Returns
        -------
        distance : npt.NDArray[np.float32]
            Distance array of shape (3,).
        """
        raise NotImplementedError()

    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        return self.distance(p) <= 0


class SDFCompose(SDF):
    """Compose multiple SDFs."""

    def __init__(self, sdfs: List[SDF]) -> None:
        self.sdfs = sdfs

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.min([sdf(p) for sdf in self.sdfs], axis=0)

    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        flags = np.stack([sdf.is_in(p) for sdf in self.sdfs])
        return np.any(flags.T, axis=1)


class SDFRoundCone(SDF):
    """Round cone is made up of two balls and a cylinder."""

    def __init__(
        self, a: npt.ArrayLike, b: npt.ArrayLike, ra: float, rb: float
    ) -> None:
        """SDF of round cone.

        Parmeters
        ---------
        a, b : ArrayLike
            Coordinates of point A/B of shape (3,).
        ra, rb : float
            Radius of point A/B.
        """

        self.a = np.array(a, dtype=np.float32)
        self.b = np.array(b, dtype=np.float32)
        self.ra = ra
        self.rb = rb

        assert tuple(self.a.shape) == (3,), "a should be vector of 3d"
        assert tuple(self.b.shape) == (3,), "b should be vector of 3d"

    def distance(self, p: npt.ArrayLike) -> npt.NDArray[np.float32]:
        p = np.array(p, dtype=np.float32)
        assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

        a = self.a
        b = self.b
        ra = self.ra
        rb = self.rb

        # sampling independent computations (only depend on shape)
        ba = b - a
        l2 = np.dot(ba, ba)
        rr = ra - rb
        a2 = l2 - rr * rr
        il2 = 1.0 / l2

        # sampling dependant computations
        pa = p - a
        y = np.dot(pa, ba)
        z = y - l2
        x = pa * l2 - np.outer(y, ba)
        x2 = np.sum(x * x, axis=1)
        y2 = y * y * l2
        z2 = z * z * l2

        # single square root!
        k = np.sign(rr) * rr * rr * x2
        dis = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - ra

        lt = np.sign(z) * a2 * z2 > k
        dis[lt] = np.sqrt(x2[lt] + z2[lt]) * il2 - rb

        rt = np.sign(y) * a2 * y2 < k
        dis[rt] = np.sqrt(x2[rt] + y2[rt]) * il2 - ra

        return dis

    def is_in(self, p: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        p = np.array(p, dtype=np.float32)
        assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

        inner = np.logical_and(
            np.all(p <= np.max([self.a + self.ra, self.b + self.rb], axis=0), axis=1),
            np.all(p >= np.min([self.a - self.ra, self.b - self.rb], axis=0), axis=1),
        )
        flags = np.full((p.shape[0]), False, dtype=np.bool_)
        flags[inner] = self.distance(p[inner]) <= 0
        return flags
