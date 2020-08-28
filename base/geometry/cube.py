import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import List, Tuple, Union, Any  # type: ignore

from geometry import box  # type: ignore

# Specification for a cube type:
# Must be a -1 x 2 x 2 numpy array of type np.float64
# First dimension represents the number of boxes in cube.
# Second dimension represents point1 (upper left) and point2 (bottom right) of each box.
# Third dimension represents the y, x coordinates of each box.


class AssertCubeError(Exception):
    def __init__(self, message):
        super(AssertCubeError, self).__init__(message)


def assert_cube(arr: NDArray[np.float64], debug: bool = False):
    try:
        assert len(arr.shape) == 3
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertCubeError(
            f"Cube array shape is invalid. Expected: 3, Actual: {len(arr.shape)}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.shape[1] == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertCubeError(
            f"Cube array dimension-1 length is invalid. Expected: 2, Actual: {arr.shape[1]}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.shape[2] == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertCubeError(
            f"Cube array dimension-2 length is invalid. Expected: 2, Actual: {arr.shape[2]}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.dtype == np.float64
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertCubeError(
            f"Cube array dtype is invalid. Expected: np.float64, Actual: {arr.dtype}"
        )
    except Exception as e:
        raise e

    try:  # TODO: Replace with numpy-optimized routine.
        for i in np.arange(0, arr.shape[0]):
            box.assert_box(arr[i, :, :], debug)
    except box.AssertBoxError as e:
        print(
            "While running box assertions during a cube assertion, the following error occurred:"
        )
        print(e.__str__())
        raise AssertCubeError("Encountered an error when checking cube boxes")
    except Exception as e:
        raise e


def new_cube(
    boxes: Union[List[NDArray[(2, 2), np.float64]], Tuple[NDArray[(2, 2), np.float64]]],
    debug: bool = False,
) -> NDArray[(-1, 2, 2), np.float64]:
    cube = np.concatenate(boxes, 0)
    assert_cube(cube, debug)
    return cube


def get_one_point1(
    cube: NDArray[(Any, 2, 2), np.float64], i: int
) -> NDArray[(2,), np.float64]:
    return box.get_point1(cube[i, :, :])


def get_point1(cube: NDArray[(Any, 2, 2), np.float64]) -> NDArray[(Any, 2), np.float64]:
    # TODO: Replace with numpy-optimized routine
    pts = np.zeros((cube.shape[0], 2))
    for i in np.arange(0, cube.shape[0]):
        pts[i] = box.get_point1(cube[i, :, :])
    return pts


def get_one_point2(
    cube: NDArray[(Any, 2, 2), np.float64], i: int
) -> NDArray[(2,), np.float64]:
    return box.get_point2(cube[i, :, :])


def get_point2(cube: NDArray[(Any, 2, 2), np.float64]) -> NDArray[(Any, 2), np.float64]:
    # TODO: Replace with numpy-optimized routine
    pts = np.zeros((cube.shape[0], 2))
    for i in np.arange(0, cube.shape[0]):
        pts[i] = box.get_point2(cube[i, :, :])
    return pts


def get_one_area(cube: NDArray[(Any, 2, 2), np.float64], i: int) -> float:
    return box.get_area(cube[i, :, :])


def get_area(cube: NDArray[(Any, 2, 2), np.float64]) -> NDArray[np.float64]:
    # TODO: Replace with numpy-optimized routine
    areas = np.zeros((cube.shape[0]))
    for i in np.arange(0, cube.shape[0]):
        areas[i] = box.get_area(cube[i, :, :])
    return areas


def one_intersection_ok(
    cube1: NDArray[(Any, 2, 2), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    i: int,
    round_decimals: int = 8,
) -> bool:
    return box.intersection_ok(cube1[i, :, :], cube2[i, :, :], round_decimals)


def intersection_ok(
    cube1: NDArray[(Any, 2, 2), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    round_decimals: int = 8,
) -> bool:
    # TODO: Replace with numpy-optimized routine
    overlaps = np.full(cube1.shape[0], False, np.bool)
    for i in np.arange(0, cube1.shape[0]):
        overlaps[i] = box.intersection_ok(
            cube1[i, :, :], cube2[i, :, :], round_decimals
        )
    return overlaps


def one_intersection(
    cube1: NDArray[(Any, 2, 2), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    i: int,
) -> float:
    return box.intersection(cube1[i, :, :], cube2[i, :, :])


def intersection(
    cube1: NDArray[(Any, 2, 2), np.float64], cube2: NDArray[(Any, 2, 2), np.float64]
) -> float:
    # TODO: Replace with numpy-optimized routine
    overlaps = np.zeros(cube1.shape[0])
    for i in np.arange(0, cube1.shape[0]):
        overlaps[i] = box.intersection(cube1[i, :, :], cube2[i, :, :])
    return overlaps


def one_union(
    cube1: NDArray[(Any, 2, 2), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    i: int,
) -> float:
    return box.union(cube1[i, :, :], cube2[i, :, :])


def union(
    cube1: NDArray[(Any, 2, 2), np.float64], cube2: NDArray[(Any, 2, 2), np.float64]
) -> float:
    # TODO: Replace with numpy-optimized routine
    unions = np.zeros(cube1.shape[0])
    for i in np.arange(0, cube1.shape[0]):
        unions[i] = box.union(cube1[i, :, :], cube2[i, :, :])
    return unions
