from typing import Optional, List
from exceptions.array_shape import WrongDimensionShapeError, WrongNumberOfDimensionsError, MismatchedFirstDimensionError
import numpy as np


class BaseNonMaximumSuppression:
    def __init__(self, iou_threshold: float, confidence_threshold: Optional[float] = None, *args, **kwargs):
        """Base class for non-maximum suppression.

        Args:
            iou_threshold: The value that intersection over union that each pair of
            predictions must exceed in order to removal a given proposal during
            non-maximum-suppression processing. Must be between 0 and 1 (inclusive).
            confidence_threshold: The value a prediction's confidence must exceed in order
            to not be filtered out before non-maximum suppression processing. Must be between
            0 and 1 (inclusive).
        """
        if iou_threshold is None:
            raise Exception("iou_threshold cannot be None")
        elif iou_threshold > 1:
            raise Exception("iou_threshold cannot be greater than 1")
        elif iou_threshold < 0:
            raise Exception("iou_threshold cannot be less than 0")
        self.iou_threshold = iou_threshold

        if confidence_threshold is None:
            confidence_threshold = 1
        elif confidence_threshold > 1:
            raise Exception("confidence threshold cannot be greater than 1")
        elif confidence_threshold < 0:
            raise Exception("confidence threshold cannot be less than 0")
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _check_n_dimensions(arr: np.ndarray, n_dimensions: int, func_name: str, arg_name: str,
                            multi_elem: Optional[int] = None):
        if multi_elem is not None:
            msg = f"for list element {multi_elem} - "
        else:
            msg = ""

        msg += f"in {func_name}() the {arg_name} argument had the wrong number of dimensions: actual={len(arr.shape)}, " \
               f"expected={n_dimensions}"
        if len(arr.shape) != n_dimensions:
            raise WrongNumberOfDimensionsError(msg)

    @staticmethod
    def _check_dimension_length(arr: np.ndarray, dimension_ix: int, dimension_length: int, func_name: str,
                                arg_name: str, multi_elem: Optional[int] = None):
        if multi_elem is not None:
            msg = f"for list element {multi_elem} - "
        else:
            msg = ""

        msg += f"in {func_name}() the {arg_name} argument had the wrong size at dimension {dimension_ix}: " \
               f"actual={arr.shape[dimension_ix]}, expected={dimension_length}"
        if arr.shape[dimension_ix] != dimension_length:
            raise WrongDimensionShapeError(msg)

    @staticmethod
    def _compare_first_dimension(arr1: np.ndarray, arr1_ix: int, arr2: np.ndarray, arr2_ix: int, func_name: str,
                                 arr1_name: str, arr2_name: str, multi_elem: Optional[int] = None):
        if multi_elem is not None:
            msg = f"for list element {multi_elem} - "
        else:
            msg = ""

        msg += f"in {func_name} the length of dimension {arr1_ix} in {arr1_name} did not equal the length of " \
               f"dimension {arr2_ix} in {arr2_name}: arr1_length: {arr1.shape[arr1_ix]}, arr2_length: " \
               f"{arr2.shape[arr2_ix]}"
        if arr1.shape[arr1_ix] != arr2.shape[arr2_ix]:
            raise MismatchedFirstDimensionError(msg)

    def transform(self, coordinates: List[np.ndarray], confidences: List[np.ndarray], *args, **kwargs) -> (np.ndarray, np.ndarray):
        """Base transform method for non-maximum suppression.

        This is a method of a base class that should be inherited when creating new
        non-maximum suppression algorithms. This method is intended to take in a list
        of image predictions (i.e. a list where each element is a full image's worth of
        predictions), and perform non-maximum suppression on it.

        Args:
            predictions: A list of full image predictions. Each element of this list is
            a list of all predictions in an image. The exact format of each element of the
            list will depend on the specific implementation of the inherited method.

        Returns:
            A single list of predictions. All of the image prediction lists inputted into
            the method are reduced down to a single list of predictions.

        """
        raise NotImplementedError
