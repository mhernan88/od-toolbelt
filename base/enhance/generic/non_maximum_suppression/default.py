from typing import Optional, List, Union
import numpy as np
from enhance.generic.non_maximum_suppression.base import BaseNonMaximumSuppression


class DefaultNonMaximumSuppression(BaseNonMaximumSuppression):
    def __init__(
        self,
        iou_threshold: float,
        confidence_threshold: Optional[float] = None,
        *args,
        **kwargs
    ):
        super().__init__(iou_threshold, confidence_threshold, *args, **kwargs)

    def multi_filter_by_confidence(
        self, coordinates: List[np.ndarray], confidences: Optional[List[np.ndarray]]
    ) -> (List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray):
        """Applies data validation and confidence_threshold filtering to multiple sets of arrays.

        Validates that arrays are the right shape, and then, if the confidences argument is not None, filters out any
        elements, from both arrays, where the confidences array element is less than self.confidence_threshold.

        Args:
            coordinates: A list of numpy arrays (length should match length of confidences argument, if not None).
                For specifications of each array, see documentation for filter_by_confidence() method of this class.
            confidences: A list of numpy arrays (length should match length of confidences argument, if not None).
                For specifications of each array, see documentation for filter_by_confidence() method of this class.

        Returns:
            coordinates: A similar list to what was passed into the function. Each element of this list has
                filter_by_confidence() applied to it. Additionally, if, after running filter_by_confidence() on a
                given list element it ends up being an empty array (i.e. the first dimension size is 0), then that list
                element is removed from the list.
            confidences: A similar list to what was passed into the function, if the passed value was not None. If the
                passed value was not None, then each element has filter_by_confidence() applied to it. Additionally, if,
                after running filter_by_confidence() on a given list element it ends up being an empty array (i.e. the
                first dimension size is 0), then that list element is removed from the list.
            coordinates_combined: A three dimensional numpy array of float type. The shape of the array is (-1, 4, 2),
                where the first dimension represents a set of coordinates, the second dimension represents each of the
                corners of a bounding box, and the second dimension represents the values of the y and x coordinates
                (the first value must be the y coordinate, and the second value must be the x coordinate). This array
                is constructed by merging all of the list elements of the coordinates return value, along their first
                dimension, into a single array.
            confidences_combined: A one-dimensional array of float type of confidence values, if the passed confidences
                argument was not None, else None. The shape of the array is (-1, 1), where the first dimension
                represents each confidence value. The second dimension is simply a place to hold this value. This array
                is constructed by merging all of the list elements of the coordinates return value, along their first
                dimension, into a single array.

        Raises:
            WrongNumberOfDimensionsError: This occurs if the number of dimensions of any element of coordinates or
                confidences (if not None) is not what was expected.
            WrongDimensionShapeError: This occurs if the shape of any element of coordinates or confidences (if not
                None) is not what was expected.
            MismatchedFirstDimensionError: This occurs if the first dimension of a given element of coordinates and the
                first dimension of a given element of confidences (with the same list index as the element from
                coordinates) have different lengths.
        """
        function_name = "multi_filter_by_confidence"

        # Coordinates is a list of multiple 3 dimensional numpy arrays.
        # Coordinates element dimension 0 = number of observations per image.
        # Coordinates element dimension 1 = number of corners (always 4).
        # Coordinates element dimension 2 = number of values per corner (always 2: y and x).
        for i, coordinates_array in enumerate(coordinates):
            self._check_n_dimensions(
                coordinates_array, 3, function_name, "coordinates", i
            )
            self._check_dimension_length(
                coordinates_array, 1, 4, function_name, "coordinates", i
            )
            self._check_dimension_length(
                coordinates_array, 2, 2, function_name, "coordinates", i
            )

        # Confidences is a list of multiple 1 dimensional numpy arrays.
        # Confidences element dimension 0 = number of observations per image.
        if confidences is not None:
            for i, confidences_array in enumerate(confidences):
                self._check_n_dimensions(
                    confidences_array, 1, function_name, "confidences", i
                )

            for i, coordinates_array, confidences_array in enumerate(
                zip(coordinates, confidences)
            ):
                self._compare_first_dimension(
                    coordinates_array,
                    0,
                    confidences_array,
                    0,
                    function_name,
                    "coordinates",
                    "confidences",
                    i,
                )
                coordinates[i], confidences[i] = self.filter_by_confidence(
                    confidences_array, confidences_array
                )
            coordinates = [c for c in coordinates if c is not None]
            confidences = [c for c in confidences if c is not None]

            coordinates_combined = np.concatenate(coordinates, axis=0)
            confidences_combined = np.concatenate(confidences, axis=0)

            coordinates_combined, confidences_combined = self.filter_by_confidence(
                coordinates_combined, confidences_combined
            )
        else:
            coordinates_combined = np.concatenate(coordinates, axis=0)
            confidences_combined = None

        return coordinates, confidences, coordinates_combined, confidences_combined

    def filter_by_confidence(
        self, coordinates: np.ndarray, confidences: Optional[np.ndarray] = None
    ) -> (np.ndarray, np.ndarray):
        """Applies data validation and confidence_threshold filtering to arrays.

        Validates that arrays are the right shape, and then, if the confidences argument is not None, filters out any
        elements, from both arrays, where the confidences array element is less than self.confidence_threshold.

        Args:
            coordinates: A three dimensional numpy array of float type. The shape of the array must be (-1, 4, 2), where
                the first dimension represents a set of coordinates, the second dimension represents each of the corners
                of a bounding box, and the second dimension represents the values of the y and x coordinates (the first
                value must be the y coordinate, and the second value must be the x coordinate). Values must be between
                0 and 1 (inclusive) and represent the relative position, to the top-left corner, of a coordinate on an
                image (i.e. a coordinate of 0.4257, 0.9511 represents a coordinate that is 42.57% of the way down from
                the top of the image and 95.11% of the way to the right of the image).
            confidences: An optional one-dimensional array of float type of confidence values. The shape of the array
                must be (-1, 1), where the first dimension represents each confidence value. The second dimension is
                simply a place to hold this value. Values must be between 0 and 1 (inclusive) and represent the level
                of confidence a prediction has.

        Returns:
            coordinates: A similar array to what was passed into the function. This array has the same restrictions on
                shape and values. The main exception is that, if the confidences argument was not None, then some slices
                of this array (along the first dimension) may be filtered out.
            confidences: A similar array to what was passed into this function, if the passed value was not None. The
                main exception is that values of this array may be filtered out. If the passed argument was None, then
                None is returned.

        Raises:
            WrongNumberOfDimensionsError: This occurs if the number of dimensions of coordinates or confidences
            (if not None) is not what was expected.
            MismatchedFirstDimensionError: This occurs if the first dimension of coordinates and the first dimension of
                confidences have different lengths.
        """
        function_name = "filter_by_confidence"

        # Coordinates dimension 0 = number of observations.
        # Coordinates dimension 1 = number of corners (always 4).
        # Coordinates dimension 2 = number of values per corner (always 2: y and x).
        self._check_n_dimensions(coordinates, 3, function_name, "coordinates")

        if confidences is not None:
            self._check_n_dimensions(confidences, 1, function_name, "confidences")
            self._compare_first_dimension(
                coordinates,
                0,
                confidences,
                0,
                function_name,
                "coordinates",
                "confidences",
            )

            confidences_to_keep = confidences > self.confidence_threshold
            confidences = confidences[confidences_to_keep]
            coordinates = coordinates[confidences_to_keep, :]

        if len(coordinates) == 0 or len(confidences) == 0:
            return None, None

        return coordinates, confidences

    def nms(self, coordinates: np.ndarray) -> (np.ndarray, np.ndarray):
        return coordinates

    def _find_overlap(
        self, coordinates1: np.ndarray, coordinates2: np.ndarray, unsafe=False
    ):
        function_name = "_find_overlap"

        if not unsafe:
            self._check_n_dimensions(coordinates1, 1, function_name, "coordinates1")
            self._check_dimension_length(
                coordinates1, 0, 4, function_name, "coordinates1"
            )

            self._check_n_dimensions(coordinates2, 1, function_name, "coordinates2")
            self._check_dimension_length(
                coordinates2, 0, 4, function_name, "coordinates2"
            )

    def transform(
        self,
        coordinates: List[np.ndarray],
        confidences: List[np.ndarray],
        *args,
        **kwargs
    ) -> (np.ndarray, np.ndarray):
        (
            coordinates,
            confidences,
            coordinates_combined,
            confidences_combined,
        ) = self.multi_filter_by_confidence(coordinates, confidences)

        coordinates_combined, confidences_combined = self.nms(coordinates_combined)
