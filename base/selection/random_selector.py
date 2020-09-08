# Copyright Michael Ayabarreno-Hernandez. All rights reserved.

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple, Dict, List  # type: ignore


from base.selection.base import Selector  # type: ignore


class RandomSelector(Selector):
    def __init__(self):
        """No configuration arguments in this class.
        """
        super().__init__()

    def select(self, bids: List[int]) -> int:
        return np.random.choice(bids)
