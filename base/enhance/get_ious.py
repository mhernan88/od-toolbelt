import itertools  # type: ignore
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple  # type: ignore

from geometry import cube  # type: ignore


def get_ious(
    cube_arr: NDArray[(Any, 2, 2), np.float64],
    confidences: NDArray[(Any,), np.float64],
    exact: bool = False,
    *args,
    **kwargs,
) -> Tuple[
    NDArray[(Any, 2, 2), np.float64],
    NDArray[(Any, 2, 2), np.float64],
    NDArray[(Any,), np.float64],
    NDArray[(Any,), np.float64],
    NDArray[(Any,), np.float64],
]:
    # For a given image (or set of images), compare each set of coordinates (np.ndarray of shape 4x2) to each
    # other set of coordinates. This is accomplished through a cartesian product of the coordinates list with
    # itself.
    cube_cartesian = itertools.product(cube_arr, cube_arr)
    confidences_cartesian = itertools.product(confidences, confidences)

    # We only want to calculate the iou for coordinate pairs that have not yet been evaluated. To accomplish that
    # we can add hashes of any already-checked coordinates to this set.
    checked_coordinates = set()

    # This output_shape is the number of unique pairs we should be evaluating.
    n = cube_arr.shape[0]
    output_shape = int((n ** 2 - n) / 2)

    # The below arrays are used to capture data from the below loop.
    arr1 = np.zeros((output_shape, cube_arr.shape[1], cube_arr.shape[2])).astype(
        np.float64
    )
    arr1_confidences = np.zeros(output_shape)
    arr2 = np.zeros((output_shape, cube_arr.shape[1], cube_arr.shape[2])).astype(
        np.float64
    )
    arr2_confidences = np.zeros(output_shape)

    i = 0
    # TODO: Replace with numpy-optimized routine.
    for box_arr, conf in zip(cube_cartesian, confidences_cartesian):
        if exact:
            if np.array_equal(box_arr[0], box_arr[1], *args, **kwargs):
                continue
        else:
            if np.allclose(box_arr[0], box_arr[1], *args, **kwargs):
                continue

        hash_prod0 = hash(box_arr[0].tobytes())
        hash_prod1 = hash(box_arr[1].tobytes())
        hash_prod = hash_prod0 * hash_prod1

        if hash_prod in checked_coordinates:
            continue
        checked_coordinates.add(hash_prod)

        arr1[i] = box_arr[0]
        arr1_confidences[i] = conf[0]
        arr2[i] = box_arr[1]
        arr2_confidences[i] = conf[1]
        i += 1

    ious = np.divide(cube.intersection(arr1, arr2), cube.union(arr1, arr2))
    return arr1, arr2, arr1_confidences, arr2_confidences, ious
