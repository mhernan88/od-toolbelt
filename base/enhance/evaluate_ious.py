import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Callable, Tuple, Dict  # type: ignore


def evaluate_ious(
    ious: NDArray[(Any,), np.float64],
    cube1: NDArray[(Any, 2, 2), np.float64],
    confidences1: NDArray[(Any,), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    confidences2: NDArray[(Any,), np.float64],
    iou_threshold: float,
    selection_func: Callable[
        [
            NDArray[(Any, 2, 2), np.float64],
            NDArray[(Any, 2, 2), np.float64],
            NDArray[(Any,), np.float64],
            NDArray[(Any,), np.float64],
            str,
            str,
            Dict[str, Any],
        ],
        Tuple[NDArray[(2, 2), np.float64], float],
    ],
    selection_kwargs: Dict[str, Any],
    method: str,
    metric: str,
    round_decimals: int = 8,
) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
    cube1 = np.round(cube1, decimals=round_decimals)
    cube2 = np.round(cube2, decimals=round_decimals)

    selected_boxes = np.zeros(cube1.shape[0], 2)
    selected_confidences = np.zeros(cube1.shape[0])

    unique_coords = np.unique(cube1, axis=0)
    for i in np.arange(0, unique_coords.shape[0]):
        # Evaluating coord1[i]
        this_unique_coord = unique_coords[i]

        # Get the indexes of coords1 that match this_unique_coord and that have an iou greater than or equal to
        # the threshold
        coords1_minus_this_unique_coord = cube1 - this_unique_coord
        coords1_different_from_this_unique_coord = np.sum(
            np.sum(coords1_minus_this_unique_coord, axis=2), axis=1
        )
        ixs_to_evaluate = np.argwhere(
            coords1_different_from_this_unique_coord == 0 and ious >= iou_threshold
        )

        # Next, pull those indexes from the first dimension of the below arrays.
        filtered_cube1 = cube1[ixs_to_evaluate, :, :]  # Should all be the same.
        filtered_confidences1 = confidences1[ixs_to_evaluate]  # Should all be the same.
        filtered_cube2 = cube2[ixs_to_evaluate, :, :]  # Should be different values.
        filtered_confidences2 = confidences2[
            ixs_to_evaluate
        ]  # Should be different values.

        selected_box, selected_confidence = selection_func(
            filtered_cube1,
            filtered_cube2,
            filtered_confidences1,
            filtered_confidences2,
            method,
            metric,
            selection_kwargs,
        )
        selected_boxes[i, :] = selected_box
        selected_confidences[i] = selected_confidence
    return selected_boxes, selected_confidences
