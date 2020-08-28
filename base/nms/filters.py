import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Optional, List, Tuple  # type: ignore


def filter_by_confidence(
    cube: NDArray[(Any, 2, 2), np.float64],
    confidences: Optional[NDArray[(Any,), np.float64]] = None,
    confidence_threshold: Optional[float] = None,
) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
    """Applies data validation and confidence_threshold filtering to arrays.

    If the confidences argument is not None, filters out any elements, from both arrays, where the confidences
    array element is less than self.confidence_threshold.

    Args:
        cube: An array that implements the requirements of geometry.cube.assert_cube(). This represents a "stack"
            of bounding box coordinates.
        confidences: An array that has the same 0th dimension size as cube. This represents the confidence of each
            prediction in the "stack".
        confidence_threshold: The level of confidence a box must exceed in order to remain in the cube.

    Returns:
        cube: An array that implements the requirements of geometry.cube.assert_cube(). The main difference between
            this and the cube argument is, if the confidences argument was not None, then some slices of this array
            (along the 0th dimension) may be filtered out.
        confidences: An array that has the same 0th dimension size as the return cube. The main difference between
            this and the confidences argument is, if the confidences argument was not None, then some values of
            this array may be filtered out. If the passed argument was None, then None is returned.
    """
    if confidences is not None:
        confidences_to_keep = confidences > confidence_threshold
        confidences = confidences[confidences_to_keep]
        cube = cube[confidences_to_keep, :, :]

    if cube.shape[0] == 0 or len(confidences) == 0:
        return None, None

    return cube, confidences


def multi_filter_by_confidence(
    cubes: List[NDArray[(Any, 2, 2), np.float64]],
    confidences: Optional[List[NDArray[(Any,), np.float64]]],
) -> Tuple[
    NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64],
]:
    """Applies data validation and confidence_threshold filtering to multiple sets of arrays.

    If the confidences argument is not None, filters out any elements, from both arrays, where the confidences
    array element is less than self.confidence_threshold.

    Args:
        cubes: A list of numpy arrays (length should match length of confidences argument, if not None). Each
            list element should implement the requirements of geometry.cube.assert_cube().
        confidences: A list of numpy arrays (length should match length of confidences argument, if not None).
            For specifications of each array, see documentation for filter_by_confidence() method of this class.

    Returns:
        cubes_combined: An array that implements the requirements of geometry.cube.assert_cube(). This is a
            filtered-down version of the cubes argument concatenated along its 0th axis.
        confidences_combined: Either None or a 1-dimensional array of confidence values. This is a filtered-down
            version of the confidences argument concatenated along its 0th axis.
    """
    if confidences is not None:
        cubes = [c for c in cubes if c is not None]
        confidences = [c for c in confidences if c is not None]

        cubes_combined = np.concatenate(cubes, axis=0)
        confidences_combined = np.concatenate(confidences, axis=0)

        cubes_combined, confidences_combined = filter_by_confidence(
            cubes_combined, confidences_combined
        )
    else:
        cubes_combined = np.concatenate(cubes, axis=0)
        confidences_combined = None

    return cubes_combined, confidences_combined
