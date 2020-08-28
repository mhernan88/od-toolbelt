import numpy as np
from nptyping import NDArray
from typing import Any, Tuple


def random_selector(
    cube1: NDArray[(Any, 2, 2), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    confidences1: NDArray[(Any,), np.float64],
    confidences2: NDArray[(Any,), np.float64],
    **kwargs
) -> Tuple[NDArray[(2, 2), np.float64], float]:
    """Chooses a random box from two cubes.

    Args:
        cube1: An array that implements the requirements of geometry.cube.assert_cube(). This represents a "stack" of
            bounding box coordinates. This values along 0th dimension of this array should all be identical. This
            represents the bounding box that we are comparing all other bounding boxes to in selection.
        cube2: An array that implements the requirements of geometry.cube.assert_cube(). This represents a "stack" of
            bounding box coordinates. This represents the bounding boxes that we are comparing to cube1.
        confidences1:
            An array that has a 0th dimension length equal to the 0th dimension length of cube1. This represents
            the confidence associated with each bounding box in cube1.
        confidences2:
            An array that has a 0th dimension length equal to the 0th dimension length of cube2. This represents
            the confidence associated with each bounding box in cube2.
        **kwargs:
            Unused kwargs to match expected function signature of enhance/evaluate_ious/evaluate_ious().

    Returns:
        selected_box: An array that implements the requirements of geometry.box.assert_box(). This represents the
            selected bounding box.
        selected_confidence: This represents the selected confidence associated with the selected bounding box.

    Raises:
        AssertionError: This occurs if the 0th dimension of cube1 and confidences1 do not match or if the 0th
            dimension of cube2 and confidences2 do not match.
    """
    assert cube1.shape[0] == confidences1.shape[0]
    assert cube2.shape[0] == confidences2.shape[0]
    cube_combined = np.array((cube2.shape[0] + 1, 2, 2), dtype=np.float64)
    cube_combined[: cube2.shape[0], :, :] = cube2
    cube_combined[-1, :, :] = cube1[0, :, :]

    confidences_combined = np.array(confidences2[0] + 1, dtype=np.float64)
    confidences_combined[: confidences2.shape[0]] = confidences2
    confidences_combined[-1] = confidences1[0]

    selected_ix = np.random.choice(np.arange(0, cube_combined.shape[0]))

    return cube_combined[selected_ix, :, :], confidences_combined[selected_ix]
