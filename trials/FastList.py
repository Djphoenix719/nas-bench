from typing import Generic
from typing import Iterable
from typing import Set
from typing import TypeVar

_T = TypeVar("_T")


class FastList(list, Generic[_T]):
    """
    A list that maintains a collection of members in a set for fast lookups.
    """

    _set: Set[_T]

    def __init__(self, seq: Iterable[_T] = ()):
        super(FastList, self).__init__(seq)
        self._set = set(seq)

    def __contains__(self, item):
        return item in self._set

    def remove(self, value: _T) -> None:
        raise Exception("FastList does not support remove.")
