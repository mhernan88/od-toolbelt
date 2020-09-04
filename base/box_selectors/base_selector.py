import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple, Dict  # type: ignore


class BaseSelector:
    def __init__(self, *args, **kwargs):
        pass

    def select(
            self,
            cube1: NDArray[(Any, 2, 2), np.float64],
            cube2: NDArray[(Any, 2, 2), np.float64],
            confs1: NDArray[(Any,), np.float64],
            confs2: NDArray[(Any,), np.float64],
            kwargs: Dict[str, Any]
    ) -> Tuple[NDArray[(2, 2), np.float64], float]:
        raise NotImplementedError
