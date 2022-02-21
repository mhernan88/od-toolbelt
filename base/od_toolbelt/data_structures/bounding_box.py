import numpy as np
import numpy.typing as npt
from numba import float32, int32
from numba.experimental import jitclass

spec = [
    ("coordinates", float32[:, :]),
    ("confidence", float32),
    ("label", int32),
    ("boxid", int32),
]


@jitclass(spec)
class BoundingBox:
    def __init__(
            self,
            coordinates: npt.NDArray[np.float32],
            confidence: float,
            label: int,
            boxid: int,
    ):
        self.coordinates = coordinates
        self.confidence = confidence
        self.label = label
        self.boxid = boxid

    def __eq__(self, other):
        return self.boxid == other.boxid

    def __gt__(self, other):
        return self.boxid > other.boxid

    def __ge__(self, other):
        return self.boxid >= other.boxid

    def __lt__(self, other):
        return self.boxid < other.boxid

    def __le__(self, other):
        return self.boxid <= other.boxid

    def __str__(self):
        return f"a bounding box with id: {self.id}"
