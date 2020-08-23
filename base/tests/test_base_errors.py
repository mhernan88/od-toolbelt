import pytest
import numpy as np
from enhance.generic.non_maximum_suppression.default import DefaultNonMaximumSuppression
from exceptions.array_shape import WrongDimensionShapeError, WrongNumberOfDimensionsError, MismatchedFirstDimensionError


def test_check_n_dimensions():
    arr_shape = (10, 1)
    arr = np.zeros(arr_shape)

    nms = DefaultNonMaximumSuppression(0, 0)
    nms._check_n_dimensions(arr, 2, "test", "test", False)
    assert True


def test_check_n_dimensions2():
    arr_shape = (10, 1)
    arr = np.zeros(arr_shape)

    nms = DefaultNonMaximumSuppression(0, 0)
    with pytest.raises(WrongNumberOfDimensionsError):
        nms._check_n_dimensions(arr, 3, "test", "test", False)


def test_check_dimension_length():
    arr_shape = (10, 1)
    arr = np.zeros(arr_shape)

    nms = DefaultNonMaximumSuppression(0, 0)
    nms._check_dimension_length(arr, 0, 10, "test", "test", False)


def test_check_dimension_length2():
    arr_shape = (10, 1)
    arr = np.zeros(arr_shape)

    nms = DefaultNonMaximumSuppression(0, 0)
    with pytest.raises(WrongDimensionShapeError):
        nms._check_dimension_length(arr, 0, 11, "test", "test", False)


def test_compare_first_dimension():
    arr1_shape = (10, 1)
    arr2_shape = (10, 1)

    arr1 = np.zeros(arr1_shape)
    arr2 = np.zeros(arr2_shape)

    nms = DefaultNonMaximumSuppression(0, 0)
    nms._compare_first_dimension(arr1, 0, arr2, 0, "test", "arr1", "arr2", False)


def test_compare_first_dimension2():
    arr1_shape = (10, 1)
    arr2_shape = (11, 1)

    arr1 = np.zeros(arr1_shape)
    arr2 = np.zeros(arr2_shape)

    nms = DefaultNonMaximumSuppression(0, 0)
    with pytest.raises(MismatchedFirstDimensionError):
        nms._compare_first_dimension(arr1, 0, arr2, 0, "test", "arr1", "arr2", False)