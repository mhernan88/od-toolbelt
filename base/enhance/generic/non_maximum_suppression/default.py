from typing import Optional, List, Union
import numpy as np
from enhance.generic.non_maximum_suppression.base import BaseNonMaximumSuppression


class DefaultNonMaximumSuppression(BaseNonMaximumSuppression):
    def __init__(self, iou_threshold: float, confidence_threshold: Optional[float] = None, *args, **kwargs):
        super().__init__(iou_threshold, confidence_threshold, *args, **kwargs)

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

            confidences_to_keep = confidences > self.confidence_threshold
            confidences = confidences[confidences_to_keep]
            coordinates = coordinates[confidences_to_keep, :]

        if len(coordinates) == 0 or len(confidences) == 0:
            return None, None

        return coordinates, confidences

    def nms(self, coordinates: np.ndarray) -> (np.ndarray, np.ndarray):
        return coordinates

    def transform(self, coordinates: List[np.ndarray], confidences: List[np.ndarray], *args, **kwargs) -> \
            (np.ndarray, np.ndarray):
        coordinates, confidences, coordinates_combined, confidences_combined = \
            self.multi_filter_by_confidence(coordinates, confidences)

        coordinates_combined, confidences_combined = self.nms(coordinates_combined)