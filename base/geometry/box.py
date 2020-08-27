import numpy as np
from nptyping import NDArray

from geometry import point

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
        raise AssertBoxError(f"Box array shape is invalid. Expected 2, Actual: {len(arr.shape)}")
    except Exception as e:
        raise e

    try:
        assert arr.shape[0] == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertBoxError(f"Box array dimension-0 length is invalid. Expected: 2, Actual: {arr.shape[0]}")
    except Exception as e:
        raise e

    try:
        assert arr.shape[1] == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertBoxError(f"Box array dimension-1 length is invalid. Expected: 2, Actual: {arr.shape[0]}")
    except Exception as e:
        raise e

    try:
        assert arr.dtype == np.float64
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertBoxError(f"Box array dtype is invalid. Expected: np.float64, Actual: {arr.dtype}")
    except Exception as e:
        raise e

    try:
        np.apply_along_axis(point.assert_point, 0, arr, debug=debug)
    except point.AssertPointError as e:
        print("While running point assertions during a box assertion, the following error occurred:")
        print(e.__str__())
        raise AssertBoxError("Encountered an error when checking box points")
    except Exception as e:
        raise e


vassert_box = np.vectorize(assert_box)


def get_point1(box: NDArray[(2, 2), np.float64]) -> NDArray[(2, ), np.float64]:
    """Gets the first (upper left) point of a box.

    Args:
        box: A "box", which is an array that meets the requirements of geometry.box.assert_box().

    Returns:
        A "point", which is an array that meets the requirements of geometry.point.assert_point().

    """
    return box[0, :].astype(np.float64)


vget_point1 = np.vectorize(get_point1, (np.float64,))


def get_point2(box: NDArray[(2, 2), np.float64]) -> NDArray[(2, ), np.float64]:
    """Gets the second (lower right) point of a box.

    Args:
        box: A "box", which is an array that meets the requirements of geometry.box.assert_box().

    Returns:
        A "point", which is an array that meets the requirements of geometry.point.assert_point().
    """
    return box[0, :].astype(np.float64)


vget_point2 = np.vectorize(get_point2, (np.float64,))


def get_area(box: NDArray[(2, 2), np.float64]) -> float:
    """Calculates the area of a box.

    Args:
        box: A "box", which is an array that meets the requirements of geometry.box.assert_box().

    Returns:

    """
    pt1 = get_point1(box)
    pt2 = get_point2(box)
    return (pt2[0] - pt1[0]) * (pt2[1] - pt1[1])


vget_area = np.vectorize(get_area, (np.float64,))


def boxes_overlap(box1: NDArray[(2, 2), np.float64], box2: NDArray[(2, 2), np.float64], round_decimals=8) -> bool:
    b1pt1 = get_point1(box1)
    b1pt2 = get_point2(box1)

    b2pt1 = get_point1(box2)
    b2pt2 = get_point2(box2)

    diff1 = np.round(np.subtract((b2pt2, b1pt1)), decimals=round_decimals)
    diff2 = np.round(np.subtract((b1pt2, b2pt1)), decimals=round_decimals)

    if np.any(diff1 <= 0) or np.any(diff2 <= 0):
        return False
    return True


vboxes_overlap = np.vectorize(boxes_overlap, (np.bool,))
