import numpy as np
from nptyping import NDArray
from typing import Any, List, Optional, Callable, Dict, Tuple

from enhance.filters import multi_filter_by_confidence
from enhance.iou import get_ious, evaluate_ious


class NonMaximumSuppression:
    def __init__(
        self,
        iou_threshold: float,
        selection_func: Callable[
            [
                NDArray[(Any, 2, 2), np.float64],
                NDArray[(Any, 2, 2), np.float64],
                NDArray[(Any,), np.float64],
                NDArray[(Any,), np.float64],
                Optional[Dict[str, Any]],
            ],
            Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any, ), np.float64]],
        ],
        selection_kwargs,
        exact,
    ):
        self.iou_threshold = iou_threshold
        self.selection_func = selection_func
        self.selection_kwargs = selection_kwargs
        self.exact = exact

    def transform(
        self,
        cubes: List[NDArray(Any, 2, 2), np.float64],
        confidences: Optional[List[NDArray[(Any,), np.float64]]],
    ) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
        cubes_filtered, confidences_filtered = multi_filter_by_confidence(
            cubes, confidences
        )
        cube1, cube2, confidences1, confidences2, ious = get_ious(
            cubes_filtered, confidences_filtered, self.exact
        )
        return evaluate_ious(
            ious,
            cube1,
            confidences1,
            cube2,
            confidences2,
            self.iou_threshold,
            self.selection_func,
            self.selection_kwargs,
        )
