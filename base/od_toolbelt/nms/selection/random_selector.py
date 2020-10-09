# Copyright 2020 Michael Hernandez

import numpy as np  # type: ignore
from typing import List  # type: ignore

from od_toolbelt.nms.selection.base import Selector  # type: ignore


class RandomSelector(Selector):
    def __init__(self):
        super().__init__()

    def select(self, bids: List[int]) -> int:
        """Selects a box at random.

        Args:
            bids: The box ids you wish to consider.

        Returns:
            A single box id from the bids argument.
        """
        return np.random.choice(bids)
