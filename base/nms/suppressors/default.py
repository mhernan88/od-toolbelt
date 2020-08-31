import logging  # type: ignore
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, List, Optional, Callable, Dict, Tuple  # type: ignore

from nms.filters import multi_filter_by_confidence  # type: ignore
from nms.iou import get_ious, evaluate_ious  # type: ignore


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
            Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]],
        ],
        logger: logging.Logger,
        confidence_threshold: float = None,
        selection_kwargs: Optional[Dict[str, Any]] = None,
        exact: bool = False,
    ):
        self.iou_threshold = iou_threshold
        self.logger = logger
        self.confidence_threshold = confidence_threshold
        self.selection_func = selection_func
        self.selection_kwargs = selection_kwargs
        self.exact = exact

    def transform(
        self,
        cubes: List[NDArray[(Any, 2, 2), np.float64]],
        confidences: Optional[List[NDArray[(Any,), np.float64]]] = None,
    ) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
        cubes_filtered, confidences_filtered = multi_filter_by_confidence(
            cubes, self.confidence_threshold, self.logger, confidences
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
