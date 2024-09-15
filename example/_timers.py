"""Dictionary-like structure with information about timers"""

# Standard library imports
import collections
import math
import statistics
from typing import TYPE_CHECKING, Any, Callable, Dict, List

# Annotate generic UserDict
if TYPE_CHECKING:
    UserDict = collections.UserDict[str, float]  # pragma: no cover
else:
    UserDict = collections.UserDict


class Timers(UserDict):
    """Custom dictionary that stores information about timers"""

    def __init__(self, *args, **kwargs):
        """Add a private dictionary keeping track of all timings"""
        super().__init__(*args, **kwargs)
        self._timings: Dict[str, List[float]] = collections.defaultdict(list)

    def add(self, name, value):
        """Add a timing value to the given timer"""
        self._timings[name].append(value)
        self.data.setdefault(name, 0)
        self.data[name] += value

    def clear(self):
        """Clear timers"""
        self.data.clear()
        self._timings.clear()

    def __setitem__(self, name, value):
        """Disallow setting of timer values"""
        raise TypeError(
            f"{self.__class__.__name__!r} does not support item assignment. "
            "Use '.add()' to update values."
        )

    def apply(self, func, name):
        """Apply a function to the results of one named timer"""
        if name in self._timings:
            return func(self._timings[name])
        raise KeyError(name)

    def count(self, name):
        """Number of timings"""
        return self.apply(len, name=name)

    def total(self, name):
        """Total time for timers"""
        return self.apply(sum, name=name)

    def min(self, name):
        """Minimal value of timings"""
        return self.apply(lambda values: min(values or [0]), name=name)

    def max(self, name):
        """Maximal value of timings"""
        return self.apply(lambda values: max(values or [0]), name=name)

    def mean(self, name):
        """Mean value of timings"""
        return self.apply(lambda values: statistics.mean(values or [0]), name=name)

    def median(self, name):
        """Median value of timings"""
        return self.apply(lambda values: statistics.median(values or [0]), name=name)

    def stdev(self, name):
        """Standard deviation of timings"""
        if name in self._timings:
            value = self._timings[name]
            return statistics.stdev(value) if len(value) >= 2 else math.nan
        raise KeyError(name)
