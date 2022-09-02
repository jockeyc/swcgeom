from typing import Any, Generic, TypeVar, cast, overload

__all__ = ["Transform", "Transforms"]

T, K = TypeVar("T"), TypeVar("K")

T1, T2, T3 = TypeVar("T1"), TypeVar("T2"), TypeVar("T3")
T4, T5, T6 = TypeVar("T4"), TypeVar("T5"), TypeVar("T6")


class Transform(Generic[T, K]):
    r"""An abstract class representing a :class:`Transform`.

    All transforms that represent a map from `T` to `K`.

    Methods
    -------
    __call__(x: T) -> K
        All subclasses should overwrite :meth:`__call__`, supporting
        applying transform in `x`.
    __repr__() -> str
        Subclasses could also optionally overwrite :meth:`__repr__`.
        If not provied, class name will be a default value.
    """

    def __call__(self, x: T) -> K:
        """Apply transform."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.__class__.__name__


class Transforms(Transform[T, K]):
    """A simple typed wrapper for transforms."""

    # fmt:off
    @overload
    def __init__(self, t1: Transform[T, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, T4],
                       t5: Transform[T4, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, T4],
                       t5: Transform[T4, T5], t6: Transform[T5, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, T4],
                       t5: Transform[T4, T5], t6: Transform[T5, T6],
                       t7: Transform[T6, Any], /, *transforms: Transform[Any, Any]) -> None: ...
    # fmt:on

    transforms: list[Transform[Any, Any]]

    def __init__(self, *transforms: Transform[Any, Any]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: T) -> K:
        """Apply transforms."""
        for transform in self.transforms:
            x = transform(x)

        return cast(K, x)

    def __getitem__(self, idx: int) -> Transform[Any, Any]:
        return self.transforms[idx]

    def __repr__(self) -> str:
        return "_".join([str(transform) for transform in self])

    def __len__(self) -> int:
        return len(self.transforms)
