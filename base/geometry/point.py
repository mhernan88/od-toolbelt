import numpy as np
from nptyping import NDArray


# Specification for a point type:
# Must be a 2-length numpy array of type np.float64.
# First dimension represents y, x coordinates.


class AssertPointError(Exception):
    def __init__(self, message):
        super(AssertPointError, self).__init__(message)


def assert_point(arr: NDArray[np.float64], debug: bool = False):
    """Asserts whether a numpy array meets the specifications of a "point" type.

    Args:
        arr: The array to evaluate.
        debug: A flag. When true, additional information is printed.

    Returns:
        None

    Raises:
        AssertionError if any of the checks fail.
    """
    try:
        assert len(arr.shape) == 1
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertPointError(
            f"Point array shape is invalid. Expected: 1, Actual: {len(arr.shape)}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.shape[0] == 2
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertPointError(
            f"Point array dimension-0 length is invalid. Expected: 2, Actual: {arr.shape[0]}"
        )
    except Exception as e:
        raise e

    try:
        assert arr.dtype == np.float64
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertPointError(
            f"Point array dtype is invalid. Expected: np.float64, Actual: {arr.dtype}"
        )
    except Exception as e:
        raise e

    try:
        assert np.all(arr >= 0)
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertPointError(
            f"Point array values were not all >= 0. Expected: all >= 0, Actual: {arr}"
        )
    except Exception as e:
        raise e

    try:
        assert np.all(arr <= 1)
    except AssertionError:
        if debug:
            print(f"Array type: {type(arr)}")
            print(f"Array values: {arr}")
        raise AssertPointError(
            f"Point array values were not all <= 1. Expected: all <= 1, Actual: {arr}"
        )
    except Exception as e:
        raise e


def new_point(y: float, x: float, debug: bool = False) -> NDArray[(2,), np.float64]:
    arr = np.array((y, x)).astype(np.float64)
    assert_point(arr, debug)
    return arr
