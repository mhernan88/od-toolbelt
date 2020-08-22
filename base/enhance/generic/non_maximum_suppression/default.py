from typing import Optional, List, Union
import numpy as np
from enhance.generic.non_maximum_suppression.base import BaseNonMaximumSuppression


class DefaultNonMaximumSuppression(BaseNonMaximumSuppression):
    def __init__(self, iou_threshold: float, confidence_threshold: Optional[float] = None, *args, **kwargs):
        super().__init__(iou_threshold, confidence_threshold, *args, **kwargs)

    @staticmethod
    def _check_n_dimensions(arr: np.ndarray, n_dimensions: int, func_name: str, arg_name: str,
                            multi_elem: Optional[int] = None):
        if multi_elem is not None:
            msg = f"for list element {multi_elem} - "
        else:
            msg = ""

        msg += f"in {func_name}() the {arg_name} argument had the wrong number of dimensions: actual={arr.shape}, " \
               f"expected={n_dimensions}"
        if len(arr.shape) != n_dimensions:
            raise Exception(msg)

    @staticmethod
    def _check_dimension_length(arr: np.ndarray, dimension_ix: int, dimension_length: int, func_name: str,
                                arg_name: str, multi_elem: Optional[int] = None):
        if multi_elem is not None:
            msg = f"for list element {multi_elem} - "
        else:
            msg = ""

        msg += f"in {func_name}() the {arg_name} argument had the wrong size at dimension {dimension_ix}: " \
               f"actual={arr.shape[dimension_ix]}, expected={dimension_length}"
        if len(arr.shape[dimension_ix] != dimension_length):
            raise Exception(msg)

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
            raise Exception(msg)

    def multi_filter_by_confidence(self, coordinates: List[np.ndarray], confidences: Optional[List[np.ndarray]]) -> \
            (List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray):
        function_name = "multi_filter_by_confidence"

        for i, coordinates_array in enumerate(coordinates):
            self._check_n_dimensions(coordinates_array, 3, function_name, "coordinates", i)
            self._check_dimension_length(coordinates_array, 1, 4, function_name, "coordinates", i)

        if confidences is not None:
            for i, confidences_array in enumerate(confidences):
                self._check_n_dimensions(confidences_array, 2, function_name, "confidences", i)

            for i, coordinates_array, confidences_array in enumerate(zip(coordinates, confidences)):
                self._compare_first_dimension(coordinates_array, 0, confidences_array, 0, function_name,
                                              "coordinates", "confidences", i)
                coordinates[i], confidences[i] = self.filter_by_confidence(confidences_array, confidences_array)
            coordinates = [c for c in coordinates if c is not None]
            confidences = [c for c in confidences if c is not None]

            coordinates_combined = np.concatenate(coordinates, axis=0)
            confidences_combined = np.concatenate(confidences, axis=0)

            coordinates_combined, confidences_combined = self.filter_by_confidence(coordinates_combined,
                                                                                   confidences_combined)
        else:
            coordinates_combined = np.concatenate(coordinates, axis=0)
            confidences_combined = None

        return coordinates, confidences, coordinates_combined, confidences_combined

    def filter_by_confidence(self, coordinates: np.ndarray, confidences: Optional[np.ndarray]) -> \
            (np.ndarray, np.ndarray):
        function_name = "filter_by_confidence"

        self._check_n_dimensions(coordinates, 2, function_name, "coordinates")

        if confidences is not None:
            self._check_n_dimensions(confidences, 1, function_name, "confidences")
            self._compare_first_dimension(coordinates, 0, confidences, 0, function_name, "coordinates", "confidences")

            confidences_to_keep = 0 if confidences < self.confidence_threshold else 1
            confidences = confidences[confidences_to_keep]
            coordinates = coordinates[confidences_to_keep, :]

        if len(coordinates) == 0 or len(confidences) == 0:
            return None, None

        return coordinates, confidences

    def transform(self, predictions: List[list], *args, **kwargs) -> list:
        pass
