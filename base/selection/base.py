import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple, Dict, List  # type: ignore


class Selector:
    def __init__(self, *args, **kwargs):
        pass

    def select(
            self, bids: List[int]
    ) -> int:
        raise NotImplementedError
