import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

from geometry import point  # type: ignore

# Specification for a box type:
# Must be a 2 x 2 numpy array of type np.float64.
# First dimension represents point1 (upper left) and point2 (bottom right).
# Second dimension represents y, x coordinates.


class AssertBoxError(Exception):
    def __init__(self, message):
        super(AssertBoxError, self).__init__(message)


def assert_box(arr: NDArray[np.float64], debug: bool = False):
    """Asserts whether a numpy array meets the specifications of a "box" type.

    Args:
        arr: The array to evaluate.

    Returns:
        None

    Raises:
        AssertBoxError: if any of the checks fail (except for the last one).
        AssertPointError: if the last check fails
    """
    try:
        assert len(arr.shape) == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertBoxError(
            f"Box array shape is invalid. Expected 2, Actual: {len(arr.shape)}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.shape[0] == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertBoxError(
            f"Box array dimension-0 length is invalid. Expected: 2, Actual: {arr.shape[0]}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.shape[1] == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertBoxError(
            f"Box array dimension-1 length is invalid. Expected: 2, Actual: {arr.shape[0]}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.dtype == np.float64
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertBoxError(
            f"Box array dtype is invalid. Expected: np.float64, Actual: {arr.dtype}"
        )
    except Exception as e:
        raise e

    try:
        np.apply_along_axis(point.assert_point, 0, arr, debug=debug)
    except point.AssertPointError as e:
        print(
            "While running point assertions during a box assertion, the following error occurred:"
        )
        print(e.__str__())
        raise AssertBoxError("Encountered an error when checking box points")
    except Exception as e:
        raise e


def new_box(
    pt1: NDArray[(2,), np.float64], pt2: NDArray[(2,), np.float64], debug: bool = False
) -> NDArray[(2, 2), np.float64]:
    box = np.array((pt1, pt2)).astype(np.float64)
    assert_box(box, debug)
    return box


def get_point1(box: NDArray[(2, 2), np.float64]) -> NDArray[(2,), np.float64]:
    """Gets the first (upper left) point of a box.

    Args:
        box: A "box", which is an array that meets the requirements of geometry.box.assert_box().

    Returns:
        A "point", which is an array that meets the requirements of geometry.point.assert_point().

    """
    return box[0, :].astype(np.float64)


def get_point2(box: NDArray[(2, 2), np.float64]) -> NDArray[(2,), np.float64]:
    """Gets the second (lower right) point of a box.

    Args:
        box: A "box", which is an array that meets the requirements of geometry.box.assert_box().

    Returns:
        A "point", which is an array that meets the requirements of geometry.point.assert_point().
    """
    return box[1, :].astype(np.float64)


def get_area(box: NDArray[(2, 2), np.float64]) -> float:
    """Calculates the area of a box.

    Args:
        box: A "box", which is an array that meets the requirements of geometry.box.assert_box().

    Returns:

    """
    pt1 = get_point1(box)
    pt2 = get_point2(box)
    return (pt2[0] - pt1[0]) * (pt2[1] - pt1[1])


def intersection_ok(
    box1: NDArray[(2, 2), np.float64],
    box2: NDArray[(2, 2), np.float64],
    round_decimals=8,
) -> bool:
    b1pt1 = get_point1(box1)
    b1pt2 = get_point2(box1)

    b2pt1 = get_point1(box2)
    b2pt2 = get_point2(box2)

    diff1 = np.round(np.subtract(b2pt2, b1pt1), decimals=round_decimals)
    diff2 = np.round(np.subtract(b1pt2, b2pt1), decimals=round_decimals)

    if np.any(diff1 <= 0) or np.any(diff2 <= 0):
        return False
    return True


def intersection(box1: NDArray[(2, 2), np.float64], box2: NDArray[(2, 2), np.float64]):
    box1_pt1 = get_point1(box1)
    box1_pt2 = get_point2(box1)
    box2_pt1 = get_point1(box2)
    box2_pt2 = get_point2(box2)

    common_pt1 = np.max((box1_pt1, box2_pt1), axis=0)
    common_pt2 = np.min((box1_pt2, box2_pt2), axis=0)

    wh = np.abs(np.subtract(common_pt2, common_pt1))
    return np.product(wh)


def union(box1: NDArray[(2, 2), np.float64], box2: NDArray[(2, 2), np.float64]):
    box1_area = get_area(box1)
    box2_area = get_area(box2)
    overlap_area = intersection(box1, box2)
    return box1_area + box2_area - overlap_area
