# type: ignore
from typing import Optional, List, Union, Any
import itertools
import numpy as np
from nptyping import NDArray

from geometry import box
from enhance.generic.non_maximum_suppression.base import BaseNonMaximumSuppression

# from enhance.generic.selection.default import DefaultSelectionAlgorithm
# from geometry import box


class DefaultNonMaximumSuppression(BaseNonMaximumSuppression):
    metric_values = ("confidence",)
    method_values = (
        "min",
        "max",
        "random",
    )

    def __init__(
        self,
        iou_threshold: float,
        confidence_threshold: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(iou_threshold, confidence_threshold, *args, **kwargs)

    def multi_filter_by_confidence(
        self,
        cubes: List[NDArray[(Any, 2, 2), np.float64]],
        confidences: Optional[List[NDArray[(Any,), np.float64]]],
    ) -> (
        List[NDArray[(Any, 2, 2), np.float64]],
        List[NDArray[(Any,), np.float64]],
        NDArray[(Any, 2, 2), np.float64],
        NDArray[(Any,), np.float64],
    ):
        """Applies data validation and confidence_threshold filtering to multiple sets of arrays.

        Validates that arrays are the right shape, and then, if the confidences argument is not None, filters out any
        elements, from both arrays, where the confidences array element is less than self.confidence_threshold.

        Args:
            cubes: A list of numpy arrays (length should match length of confidences argument, if not None). Each
                list element should implement the requirements of geometry.cube.assert_cube().
            confidences: A list of numpy arrays (length should match length of confidences argument, if not None).
                For specifications of each array, see documentation for filter_by_confidence() method of this class.

        Returns:
            cubes: A similar list to what was passed into the function. Each element of this list has
                filter_by_confidence() applied to it. Additionally, if, after running filter_by_confidence() on a
                given list element it ends up being an empty array (i.e. the first dimension size is 0), then that list
                element is removed from the list.
            confidences: A similar list to what was passed into the function, if the passed value was not None. If the
                passed value was not None, then each element has filter_by_confidence() applied to it. Additionally, if,
                after running filter_by_confidence() on a given list element it ends up being an empty array (i.e. the
                first dimension size is 0), then that list element is removed from the list.
            cubes_combined: An array that implements the requirements of geometry.cube.assert_cube(). This is the cubes
                return value concatenated along its 0th axis.
            confidences_combined: Either None or a 1-dimensional array of confidence values. This is the confidences
                return value concatenated along its 0th axis.
        """
        if confidences is not None:
            cubes = [c for c in cubes if c is not None]
            confidences = [c for c in confidences if c is not None]

            cubes_combined = np.concatenate(cubes, axis=0)
            confidences_combined = np.concatenate(confidences, axis=0)

            cubes_combined, confidences_combined = self.filter_by_confidence(
                cubes_combined, confidences_combined
            )
        else:
            cubes_combined = np.concatenate(cubes, axis=0)
            confidences_combined = None

        return cubes, confidences, cubes_combined, confidences_combined

    def filter_by_confidence(
        self,
        cube: NDArray[(Any, 2, 2), np.float64],
        confidences: Optional[NDArray[(Any,), np.float64]] = None,
    ) -> (NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]):
        """Applies data validation and confidence_threshold filtering to arrays.

        If the confidences argument is not None, filters out any elements, from both arrays, where the confidences
        array element is less than self.confidence_threshold.

        Args:
            cube: An array that implements the requirements of geometry.cube.assert_cube(). This represents a "stack"
                of bounding box coordinates.
            confidences: An array that has the same 0th dimension size as cube. This represents the confidence of each
                prediction in the "stack".

        Returns:
            cube: An array that implements the requirements of geometry.cube.assert_cube(). The main difference between
                this and the cube argument is, if the confidences argument was not None, then some slices of this array
                (along the 0th dimension) may be filtered out.
            confidences: An array that has the same 0th dimension size as the return cube. The main difference between
                this and the confidences argument is, if the confidences argument was not None, then some values of
                this array may be filtered out. If the passed argument was None, then None is returned.
        """
        if confidences is not None:
            confidences_to_keep = confidences > self.confidence_threshold
            confidences = confidences[confidences_to_keep]
            cube = cube[confidences_to_keep, :, :]

        if cube.shape[0] == 0 or len(confidences) == 0:
            return None, None

        return cube, confidences

    def get_ious(
        self,
        cube: NDArray[(Any, 2, 2), np.float64],
        confidences: NDArray[(Any,), np.float64],
        exact: bool = False,
        *args,
        **kwargs,
    ) -> (np.ndarray, np.ndarray):

        # For a given image (or set of images), compare each set of coordinates (np.ndarray of shape 4x2) to each
        # other set of coordinates. This is accomplished through a cartesian product of the coordinates list with
        # itself.
        cube_cartesian = itertools.product(cube, cube)
        confidences_cartesian = itertools.product(confidences, confidences)

        # We only want to calculate the iou for coordinate pairs that have not yet been evaluated. To accomplish that
        # we can add hashes of any already-checked coordinates to this set.
        checked_coordinates = set()

        # This output_shape is the number of unique pairs we should be evaluating.
        n = cube.shape[0]
        output_shape = int((n ** 2 - n) / 2)

        # The below arrays are used to capture data from the below loop.
        ious = np.zeros(output_shape)
        arr1 = np.zeros((output_shape, cube.shape[1], cube.shape[2]))
        arr1_confidences = np.zeros(output_shape)
        arr2 = np.zeros((output_shape, cube.shape[1], cube.shape[2]))
        arr2_confidences = np.zeros(output_shape)

        i = 0
        # Enhancement: Vectorize this loop - or move to Cython.
        for box, conf in zip(cube_cartesian, confidences_cartesian):
            if exact:
                if np.array_equal(box[0], box[1], *args, **kwargs):
                    continue
            else:
                if np.allclose(box[0], box[1], *args, **kwargs):
                    continue

            hash_prod0 = hash(box[0].tobytes())
            hash_prod1 = hash(box[1].tobytes())
            hash_prod = hash_prod0 * hash_prod1

            if hash_prod in checked_coordinates:
                continue
            checked_coordinates.add(hash_prod)

            if not box.boxes_overlap(box[0], box[1]):
                ious[i] = 0.0
            else:
                pass
                # ious[i] = self._get_ious(
                #     np.expand_dims(coord_prod[0], 0), np.expand_dims(coord_prod[1], 0)
                # )

            arr1[i] = box[0]
            arr1_confidences[i] = conf[0]
            arr2[i] = box[1]
            arr2_confidences[i] = conf[1]
            i += 1
        return arr1, arr2, arr1_confidences, arr2_confidences, ious
        # self._evaluate_ious(
        #     ious, arr1, arr1_confidences, arr2, arr2_confidences, "first"
        # )


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
