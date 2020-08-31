import logging  # type: ignore
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Optional, List, Tuple  # type: ignore


def filter_by_confidence(
    cube: NDArray[(Any, 2, 2), np.float64],
    logger: logging.Logger,
    confidences: Optional[NDArray[(Any,), np.float64]] = None,
    confidence_threshold: Optional[float] = None,
) -> Tuple[Optional[NDArray[(Any, 2, 2), np.float64]], Optional[NDArray[(Any,), np.float64]]]:
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
    f = "filter_by_confidence()"
    if confidences is not None and confidence_threshold is not None:
        logger.debug(f"in {f}, applying confidences filter (with confidence threshold of {confidence_threshold})")
        confidences_to_keep = confidences > confidence_threshold
        confidences = confidences[confidences_to_keep]
        cube = cube[confidences_to_keep, :, :]
    else:
        logger.debug(f"in {f}, skipping confidences filter")

    if cube.shape[0] == 0 or len(confidences) == 0:
        logger.debug(f"in {f}, returning None - length of cube was {cube.shape[0]} and length of confidences was {len(confidences)}")
        return None, None

    logger.debug(f"in {f}, returning values - length of cube was {cube.shape[0]} and length of confidences was {len(confidences)}")
    return cube, confidences


def multi_filter_by_confidence(
    cubes: List[NDArray[(Any, 2, 2), np.float64]],
    confidence_threshold: float,
    logger: logging.Logger,
    confidences: Optional[List[NDArray[(Any,), np.float64]]]
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
    f = "multi_filter_by_confidence()"

    if confidences is not None:
        total_boxes = sum([c.shape[0] for c in cubes])
        logger.debug(f"in {f}, applying confidence filtering - {len(cubes)} cubes (with {total_boxes} boxes) before filter and {len(confidences)} confidences before filter")

        cubes_nf = []
        confidences_nf = []

        # TODO: Optimize loop
        for i, _ in enumerate(cubes):
            if cubes[i] is not None:
                cubes_nf.append(cubes[i])
                confidences_nf.append(confidences[i])
        total_boxes = sum([c.shape[0] for c in cubes])
        logger.debug(f"in {f}, after none filtering - {len(cubes)} cubes (with {total_boxes} boxes) remain and {len(confidences)} confidences remain")

        cubes_sf = []
        confidences_sf = []

        # TODO: Optimize loop
        for i, _ in enumerate(cubes_nf):
            if cubes_nf[i].shape[0] != 0:
                cubes_sf.append(cubes_nf[i])
                confidences_sf.append(confidences_nf[i])
        total_boxes = sum([c.shape[0] for c in cubes])
        logger.debug(f"in {f}, after zero filtering - {len(cubes)} cubes (with {total_boxes} boxes) remain and {len(confidences)} confidences remain")

        cubes_combined = np.stack(cubes_sf)
        confidences_combined = np.asarray([x if x is not None else 0 for x in confidences_sf], dtype=np.float64)

        cubes_combined, confidences_combined = filter_by_confidence(
            cubes_combined, logger, confidences_combined, confidence_threshold
        )

        logger.debug(f"in {f}, after confidence filtering - {cubes_combined.shape[0]} cubes remain")
    else:
        logger.debug(f"in {f}, skipping confidence filtering")
        cubes_combined = np.concatenate(cubes, axis=0)
        confidences_combined = None

    return cubes_combined, confidences_combined
