# from typing import Optional, List, Union
# import itertools
# import numpy as np
# from enhance.generic.non_maximum_suppression.base import BaseNonMaximumSuppression
# from enhance.generic.selection.default import DefaultSelectionAlgorithm
# from geometry import box
#
#
# class DefaultNonMaximumSuppression(BaseNonMaximumSuppression):
#     method_values = ("first", "last", "random", "confidence")
#
#     def __init__(
#         self,
#         iou_threshold: float,
#         confidence_threshold: Optional[float] = None,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(iou_threshold, confidence_threshold, *args, **kwargs)
#
#     def multi_filter_by_confidence(
#         self, coordinates: List[np.ndarray], confidences: Optional[List[np.ndarray]]
#     ) -> (List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray):
#         """Applies data validation and confidence_threshold filtering to multiple sets of arrays.
#
#         Validates that arrays are the right shape, and then, if the confidences argument is not None, filters out any
#         elements, from both arrays, where the confidences array element is less than self.confidence_threshold.
#
#         Args:
#             coordinates: A list of numpy arrays (length should match length of confidences argument, if not None).
#                 For specifications of each array, see documentation for filter_by_confidence() method of this class.
#             confidences: A list of numpy arrays (length should match length of confidences argument, if not None).
#                 For specifications of each array, see documentation for filter_by_confidence() method of this class.
#
#         Returns:
#             coordinates: A similar list to what was passed into the function. Each element of this list has
#                 filter_by_confidence() applied to it. Additionally, if, after running filter_by_confidence() on a
#                 given list element it ends up being an empty array (i.e. the first dimension size is 0), then that list
#                 element is removed from the list.
#             confidences: A similar list to what was passed into the function, if the passed value was not None. If the
#                 passed value was not None, then each element has filter_by_confidence() applied to it. Additionally, if,
#                 after running filter_by_confidence() on a given list element it ends up being an empty array (i.e. the
#                 first dimension size is 0), then that list element is removed from the list.
#             coordinates_combined: A three dimensional numpy array of float type. The shape of the array is (-1, 4, 2),
#                 where the first dimension represents a set of coordinates, the second dimension represents each of the
#                 corners of a bounding box, and the second dimension represents the values of the y and x coordinates
#                 (the first value must be the y coordinate, and the second value must be the x coordinate). This array
#                 is constructed by merging all of the list elements of the coordinates return value, along their first
#                 dimension, into a single array.
#             confidences_combined: A one-dimensional array of float type of confidence values, if the passed confidences
#                 argument was not None, else None. The shape of the array is (-1, 1), where the first dimension
#                 represents each confidence value. The second dimension is simply a place to hold this value. This array
#                 is constructed by merging all of the list elements of the coordinates return value, along their first
#                 dimension, into a single array.
#
#         Raises:
#             WrongNumberOfDimensionsError: This occurs if the number of dimensions of any element of coordinates or
#                 confidences (if not None) is not what was expected.
#             WrongDimensionShapeError: This occurs if the shape of any element of coordinates or confidences (if not
#                 None) is not what was expected.
#             MismatchedFirstDimensionError: This occurs if the first dimension of a given element of coordinates and the
#                 first dimension of a given element of confidences (with the same list index as the element from
#                 coordinates) have different lengths.
#         """
#         function_name = "multi_filter_by_confidence"
#
#         # Coordinates is a list of multiple 3 dimensional numpy arrays.
#         # Coordinates element dimension 0 = number of observations per image.
#         # Coordinates element dimension 1 = number of corners (always 4).
#         # Coordinates element dimension 2 = number of values per corner (always 2: y and x).
#         for i, coordinates_array in enumerate(coordinates):
#             self._check_n_dimensions(
#                 coordinates_array, 3, function_name, "coordinates", i
#             )
#             self._check_dimension_length(
#                 coordinates_array, 1, 4, function_name, "coordinates", i
#             )
#             self._check_dimension_length(
#                 coordinates_array, 2, 2, function_name, "coordinates", i
#             )
#
#         # Confidences is a list of multiple 1 dimensional numpy arrays.
#         # Confidences element dimension 0 = number of observations per image.
#         if confidences is not None:
#             for i, confidences_array in enumerate(confidences):
#                 self._check_n_dimensions(
#                     confidences_array, 1, function_name, "confidences", i
#                 )
#
#             for i, coordinates_array, confidences_array in enumerate(
#                 zip(coordinates, confidences)
#             ):
#                 self._compare_first_dimension(
#                     coordinates_array,
#                     0,
#                     confidences_array,
#                     0,
#                     function_name,
#                     "coordinates",
#                     "confidences",
#                     i,
#                 )
#                 coordinates[i], confidences[i] = self.filter_by_confidence(
#                     confidences_array, confidences_array
#                 )
#             coordinates = [c for c in coordinates if c is not None]
#             confidences = [c for c in confidences if c is not None]
#
#             coordinates_combined = np.concatenate(coordinates, axis=0)
#             confidences_combined = np.concatenate(confidences, axis=0)
#
#             coordinates_combined, confidences_combined = self.filter_by_confidence(
#                 coordinates_combined, confidences_combined
#             )
#         else:
#             coordinates_combined = np.concatenate(coordinates, axis=0)
#             confidences_combined = None
#
#         return coordinates, confidences, coordinates_combined, confidences_combined
#
#     def filter_by_confidence(
#         self, coordinates: np.ndarray, confidences: Optional[np.ndarray] = None
#     ) -> (np.ndarray, np.ndarray):
#         """Applies data validation and confidence_threshold filtering to arrays.
#
#         Validates that arrays are the right shape, and then, if the confidences argument is not None, filters out any
#         elements, from both arrays, where the confidences array element is less than self.confidence_threshold.
#
#         Args:
#             coordinates: A three dimensional numpy array of float type. The shape of the array must be (-1, 4, 2), where
#                 the first dimension represents a set of coordinates, the second dimension represents each of the corners
#                 of a bounding box, and the second dimension represents the values of the y and x coordinates (the first
#                 value must be the y coordinate, and the second value must be the x coordinate). Values must be between
#                 0 and 1 (inclusive) and represent the relative position, to the top-left corner, of a coordinate on an
#                 image (i.e. a coordinate of 0.4257, 0.9511 represents a coordinate that is 42.57% of the way down from
#                 the top of the image and 95.11% of the way to the right of the image).
#             confidences: An optional one-dimensional array of float type of confidence values. The shape of the array
#                 must be (-1, 1), where the first dimension represents each confidence value. The second dimension is
#                 simply a place to hold this value. Values must be between 0 and 1 (inclusive) and represent the level
#                 of confidence a prediction has.
#
#         Returns:
#             coordinates: A similar array to what was passed into the function. This array has the same restrictions on
#                 shape and values. The main exception is that, if the confidences argument was not None, then some slices
#                 of this array (along the first dimension) may be filtered out.
#             confidences: A similar array to what was passed into this function, if the passed value was not None. The
#                 main exception is that values of this array may be filtered out. If the passed argument was None, then
#                 None is returned.
#
#         Raises:
#             WrongNumberOfDimensionsError: This occurs if the number of dimensions of coordinates or confidences
#             (if not None) is not what was expected.
#             MismatchedFirstDimensionError: This occurs if the first dimension of coordinates and the first dimension of
#                 confidences have different lengths.
#         """
#         function_name = "filter_by_confidence"
#
#         # Coordinates dimension 0 = number of observations.
#         # Coordinates dimension 1 = number of corners (always 4).
#         # Coordinates dimension 2 = number of values per corner (always 2: y and x).
#         self._check_n_dimensions(coordinates, 3, function_name, "coordinates")
#
#         if confidences is not None:
#             self._check_n_dimensions(confidences, 1, function_name, "confidences")
#             self._compare_first_dimension(
#                 coordinates,
#                 0,
#                 confidences,
#                 0,
#                 function_name,
#                 "coordinates",
#                 "confidences",
#             )
#
#             confidences_to_keep = confidences > self.confidence_threshold
#             confidences = confidences[confidences_to_keep]
#             coordinates = coordinates[confidences_to_keep, :]
#
#         if len(coordinates) == 0 or len(confidences) == 0:
#             return None, None
#
#         return coordinates, confidences
#
#     def nms(
#         self,
#         coordinates: np.ndarray,
#         confidences: np.ndarray,
#         exact: bool = False,
#         *args,
#         **kwargs,
#     ) -> (np.ndarray, np.ndarray):
#
#         # For a given image (or set of images), compare each set of coordinates (np.ndarray of shape 4x2) to each
#         # other set of coordinates. This is accomplished through a cartesian product of the coordinates list with
#         # itself.
#         coordinates_cartesian = itertools.product(coordinates, coordinates)
#         confidences_cartesian = itertools.product(confidences, confidences)
#
#         # We only want to calculate the iou for coordinate pairs that have not yet been evaluated. To accomplish that
#         # we can add hashes of any already-checked coordinates to this set.
#         checked_coordinates = set()
#
#         n = coordinates.shape[0]
#
#         # This output_shape is the number of unique pairs we should be evaluating.
#         output_shape = int((n ** 2 - n) / 2)
#
#         # The below arrays are used to capture data from the below loop.
#         ious = np.zeros(output_shape)
#         arr1 = np.zeros((output_shape, coordinates.shape[1], coordinates.shape[2]))
#         arr1_confidences = np.zeros(output_shape)
#         arr2 = np.zeros((output_shape, coordinates.shape[1], coordinates.shape[2]))
#         arr2_confidences = np.zeros(output_shape)
#
#         i = 0
#         # Enhancement: Vectorize this loop - or move to Cython.
#         for coord_prod, conf_prod in zip(coordinates_cartesian, confidences_cartesian):
#             if exact:
#                 if np.array_equal(coord_prod[0], coord_prod[1], *args, **kwargs):
#                     continue
#             else:
#                 if np.allclose(coord_prod[0], coord_prod[1], *args, **kwargs):
#                     continue
#
#             hash_prod0 = hash(coord_prod[0].tobytes())
#             hash_prod1 = hash(coord_prod[1].tobytes())
#             hash_prod = hash_prod0 * hash_prod1
#
#             if hash_prod in checked_coordinates:
#                 continue
#             checked_coordinates.add(hash_prod)
#
#             if not self._overlap(coord_prod[0], coord_prod[1]):
#                 ious[i] = 0.0
#             else:
#                 ious[i] = self._get_ious(
#                     np.expand_dims(coord_prod[0], 0), np.expand_dims(coord_prod[1], 0)
#                 )
#
#             arr1[i] = coord_prod[0]
#             arr1_confidences[i] = conf_prod[0]
#             arr2[i] = coord_prod[1]
#             arr2_confidences[i] = conf_prod[1]
#             i += 1
#
#             self._evaluate_ious(
#                 ious, arr1, arr1_confidences, arr2, arr2_confidences, "first"
#             )
#
#         # TODO: Evaluate each box, evaluate which other boxes exceed IOU. Throw out all boxes, except for the most
#         # TODO: confident where IOU > IOU_threshold
#
#     def _evaluate_ious(
#         self,
#         ious,
#         coords1,
#         confidences1,
#         coords2,
#         confidences2,
#         method,
#         round_decimals=8,
#         unsafe=False,
#     ):
#         function_name = "_evaluate_ious"
#         if not unsafe:
#             self._check_n_dimensions(ious, 1, function_name, "ious")
#             self._check_n_dimensions(coords1, 3, function_name, "coords1")
#             self._check_n_dimensions(confidences1, 1, function_name, "confidences1")
#             self._check_n_dimensions(coords2, 3, function_name, "coords2")
#             self._check_n_dimensions(confidences2, 1, function_name, "confidences2")
#
#             self._compare_first_dimension(
#                 ious, 0, coords1, 0, function_name, "ious", "coords1"
#             )
#             self._compare_first_dimension(
#                 ious, 0, confidences1, 0, function_name, "ious", "confidences1"
#             )
#             self._compare_first_dimension(
#                 ious, 0, coords2, 0, function_name, "ious", "coords2"
#             )
#             self._compare_first_dimension(
#                 ious, 0, confidences2, 0, function_name, "ious", "confidences2"
#             )
#             if method not in self.method_values:
#                 method_values_str = "', '".join(self.method_values)
#                 method_values_str = f"'{method_values_str}'"
#                 raise ValueError(
#                     f"in {function_name}, the method argument must be one of: {method_values_str}"
#                 )
#
#         # 1. Iterate over each unique coord1.
#         # 2. Pull all IOUs where coord1 value == unique coord1 value AND IOU > IOU threshold.
#         # 3. Apply first, last, random, confidence logic. Result should be indexes to be deleted (from rejected boxes).
#         # 4. Repeat for each value of coord1.
#
#         coords1 = np.round(coords1, decimals=round_decimals)
#         coords2 = np.round(coords2, decimals=round_decimals)
#
#         unique_coords = np.unique(coords1, axis=0)
#         for i in np.arange(0, unique_coords.shape[0]):
#             # Evaluating coord1[i]
#             this_unique_coord = unique_coords[i]
#
#             # Get the indexes of coords1 that match this_unique_coord and that have an iou greater than or equal to
#             # the threshold
#             coords1_minus_this_unique_coord = coords1 - this_unique_coord
#             coords1_different_from_this_unique_coord = np.sum(
#                 np.sum(coords1_minus_this_unique_coord, axis=2), axis=1
#             )
#             ixs_to_evaluate = np.argwhere(
#                 coords1_different_from_this_unique_coord == 0
#                 and ious >= self.iou_threshold
#             )
#
#             # Next, pull those indexes from the first dimension of the below arrays.
#             # this_ious = ious[coords1_same_as_this_unique_coord_ix]
#             this_coords1 = coords1[ixs_to_evaluate, :, :]  # Should all be the same.
#             this_confidences1 = confidences1[ixs_to_evaluate]  # Should all be the same.
#             this_coords2 = coords2[ixs_to_evaluate, :, :]  # Should be different values.
#             this_confidences2 = confidences2[
#                 ixs_to_evaluate
#             ]  # Should be different values.
#
#             selection_algorithm = DefaultSelectionAlgorithm(
#                 this_coords1, this_coords2, this_confidences1, this_confidences2
#             )
#             selected_coord, selected_confidence = selection_algorithm.dispatch(method)
#
#             print(coords1_different_from_this_unique_coord)
#             print(this_coords1)
#             print("END COORDS")
#
#     # def _get_ious(self, box: np.ndarray, coordinates2: np.ndarray, unsafe=False):
#     #     function_name = "_find_overlap"
#     #
#     #     # Coordinates dimension 0 = number of observations.
#     #     # Coordinates dimension 1 = number of corners (always 4).
#     #     # Coordinates dimension 2 = number of values per corner (always 2: y and x).
#     #
#     #     if not unsafe:
#     #         self._check_n_dimensions(coordinates1, 3, function_name, "coordinates1")
#     #         self._check_dimension_length(
#     #             coordinates1, 1, 4, function_name, "coordinates1"
#     #         )
#     #         self._check_dimension_length(
#     #             coordinates1, 2, 2, function_name, "coordinates1"
#     #         )
#     #
#     #         self._check_n_dimensions(coordinates2, 3, function_name, "coordinates2")
#     #         self._check_dimension_length(
#     #             coordinates2, 1, 4, function_name, "coordinates2"
#     #         )
#     #         self._check_dimension_length(
#     #             coordinates2, 2, 2, function_name, "coordinates2"
#     #         )
#     #
#     #     maxs = np.maximum(coordinates1, coordinates2)
#     #     mins = np.minimum(coordinates1, coordinates2)
#     #
#     #     intersection = np.zeros(coordinates1.shape)
#     #     intersection[:, 0, :] = maxs[:, 0, :]  # Upper left Y/X
#     #     intersection[:, 1, 0] = maxs[:, 1, 0]  # Upper right Y
#     #     intersection[:, 1, 1] = mins[:, 1, 1]  # Upper right X
#     #     intersection[:, 2, 0] = mins[:, 2, 0]  # Lower left Y
#     #     intersection[:, 2, 1] = maxs[:, 2, 1]  # Lower left X
#     #     intersection[:, 3, :] = mins[:, 3, :]  # Lower right Y/X
#     #
#     #     coordinates1_areas = box.get_area(coordinates1)
#     #     coordinates2_areas = box.get_area(coordinates2)
#     #     intersection_areas = box.get_area(intersection)
#     #
#     #     ious_num = np.subtract(
#     #         np.add(coordinates1_areas, coordinates2_areas), intersection_areas
#     #     )
#     #     ious = np.divide(intersection_areas, ious_num)
#     #     return ious
#     #
#     # def transform(
#     #     self,
#     #     coordinates: List[np.ndarray],
#     #     confidences: List[np.ndarray],
#     #     *args,
#     #     **kwargs,
#     # ) -> (np.ndarray, np.ndarray):
#     #     (
#     #         coordinates,
#     #         confidences,
#     #         coordinates_combined,
#     #         confidences_combined,
#     #     ) = self.multi_filter_by_confidence(coordinates, confidences)
#     #
#     #     coordinates_combined, confidences_combined = self.nms(coordinates_combined)
