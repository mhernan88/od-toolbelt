import itertools  # type: ignore
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple, Callable, Dict, Optional  # type: ignore

from geometry import cube  # type: ignore


class DefaultNonMaximumSuppression:
    i = 0
    checked_boxes = set()
    selected_boxes = None
    selected_confs = None
    selected_hash = set()

    def __init__(self, iou_threshold: float, selection_func, selection_kwargs: Dict[str, Any] = None, decimals: int = 8, exact: bool = False):
        self.exact = exact  # Whether to compare arrays with exact precision or not.
        self.decimals = decimals
        self.iou_threshold = iou_threshold

        self.selection_func = selection_func
        self.selection_kwargs = selection_kwargs

    def transform(self, c, conf):
        c = np.round(c, self.decimals)
        conf = np.round(conf, self.decimals)
        n = c.shape[0]
        output_shape = int((n ** 2 - n) / 2)

        arr1 = np.zeros((output_shape, c.shape[1], c.shape[2]), dtype=np.float64)  # The
        arr1_confs = np.zeros(output_shape)
        arr2 = np.zeros((output_shape, c.shape[1], c.shape[2]), dtype=np.float64)
        arr2_confs = np.zeros(output_shape)

        cube_cart = itertools.product(c, c)
        conf_cart = itertools.product(conf, conf)

        self.selected_boxes = np.zeros(c.shape)
        self.selected_confs = np.zeros(c.shape[0])

        i = 0
        for b, c in zip(cube_cart, conf_cart):
            arr1, arr1_confs, arr2, arr2_confs, i = self._process_box(i, b, c, arr1, arr1_confs, arr2, arr2_confs)
        arr1 = np.round(arr1, decimals=self.decimals)
        arr2 = np.round(arr2, decimals=self.decimals)
        ious = np.divide(cube.intersection(arr1, arr2), cube.union(arr1, arr2))

        unique_boxes = np.unique(arr1, axis=0)
        for i in np.arange(0, unique_boxes.shape[0]):
            self._select_box(unique_boxes[i, :, :], arr1, arr1_confs, arr2, arr2_confs, ious, i)
        self.selected_boxes = self.selected_boxes[self.selected_confs > 0, :, :]
        self.selected_confs = self.selected_confs[self.selected_confs > 0]

        return self.selected_boxes, self.selected_confs

    def _filter_used_boxes(self, cube1, conf1, cube2, conf2):
        # TODO: Optimize this loop.
        # Here, we're skipping evaluating any boxes that we've already selected.
        ixs = []
        if cube2.shape[0] > 0:
            for j in np.arange(0, cube2.shape[0]):
                fc_hash = hash(cube2[j, :, :].tobytes())
                if fc_hash not in self.selected_hash:
                    ixs.append(j)
                self.selected_hash.add(fc_hash)
        if cube1.shape[0] > 0:
            fc_hash = hash(cube1[0, :, :].tobytes())
            if fc_hash in self.selected_hash:
                cube1 = cube1[1 == 2, :, :]
                conf1 = conf1[1 == 2]
            self.selected_hash.add(fc_hash)

        filtered_cube2 = cube2[ixs, :, :]
        filtered_confidences2 = conf2[ixs]
        return cube1, conf1, filtered_cube2, filtered_confidences2

    def _process_box(
            self,
            i: int,
            b: Tuple[NDArray[(2, 2), np.float64], NDArray[(2, 2), np.float64]],
            c: Tuple[float, float],
            arr1: NDArray[(Any, 2, 2), np.float64],
            arr1_confs: NDArray[(Any,), np.float64],
            arr2: NDArray[(Any, 2, 2), np.float64],
            arr2_confs: NDArray[(Any,), np.float64],
            *args, **kwargs
    ) -> Tuple[NDArray[(Any, 2, 2), np.float64],
        NDArray[(Any,), np.float64],
        NDArray[(Any, 2, 2), np.float64],
        NDArray[(Any,), np.float64],
        int]:
        if self.exact:
            if np.array_equal(b[0], b[1], *args, **kwargs):
                return arr1, arr1_confs, arr2, arr2_confs, i
        else:
            if np.allclose(b[0], b[1], *args, **kwargs):
                return arr1, arr1_confs, arr2, arr2_confs, i

        hash_prod0 = hash(b[0].tobytes())
        hash_prod1 = hash(b[1].tobytes())
        hash_prod = hash_prod0 * hash_prod1

        if hash_prod in self.checked_boxes:
            return arr1, arr1_confs, arr2, arr2_confs, i

        self.checked_boxes.add(hash_prod)

        arr1[i] = b[0]
        arr1_confs[i] = c[0]
        arr2[i] = b[1]
        arr2_confs[i] = c[1]

        return arr1, arr1_confs, arr2, arr2_confs, i+1

    def _select_box(
            self,
            b: NDArray[(2, 2), np.float64],
            arr1: NDArray[(Any, 2, 2), np.float64],
            arr1_confs: NDArray[(Any,), np.float64],
            arr2: NDArray[(Any, 2, 2), np.float64],
            arr2_confs: NDArray[(Any,), np.float64],
            ious: NDArray[(Any,), np.float64],
            i: int
    ):
        b_diff = arr1 - b
        is_b_diff = np.sum(np.sum(b_diff, axis=2), axis=1)
        ixs = np.argwhere(np.logical_and((is_b_diff == 0), (ious > self.iou_threshold)))

        filtered_cube1 = arr1[ixs, :, :]  # Should all be the same.
        filtered_confidences1 = arr1_confs[ixs]  # Should all be the same.
        filtered_cube2 = arr2[ixs, :, :]  # Should be different values.
        filtered_confidences2 = arr2_confs[ixs]  # Should be different values.

        # filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2, selected_boxes_hash = filter_used_boxes(filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2)
        filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2 = self._filter_used_boxes(filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2)

        # TODO: TEST
        if filtered_cube1.shape[0] == 0:
            return
        # TODO: END_TEST

        filtered_cube1x = filtered_cube1[0, 0, :, :]
        filtered_cube2x = filtered_cube2[:, 0, :, :]
        filtered_confidences1x = float(filtered_confidences1.ravel()[0]),
        if isinstance(filtered_confidences1x, tuple):
            filtered_confidences1x = float(filtered_confidences1x[0])
        filtered_confidences2x = filtered_confidences2.ravel(),
        kwargsx = self.selection_kwargs

        print(f"FILTERED_CONF1: {filtered_confidences1x}")
        selected_box, selected_conf = self.selection_func.select(
            filtered_cube1x, filtered_cube2x, filtered_confidences1x, filtered_confidences2x, kwargsx
        )

        # selected_box, selected_conf = self.selection_func.select(
        #     filtered_cube1[0, 0, :, :],
        #     filtered_cube2[:, 0, :, :],
        #     filtered_confidences1.ravel()[0],
        #     filtered_confidences2.ravel(),
        #     self.selection_kwargs,
        # )

        if selected_box.shape[0] > 0:
            self.selected_boxes[i, :, :] = selected_box
            self.selected_confs[i] = selected_conf


def get_ious(
    cube_arr: NDArray[(Any, 2, 2), np.float64],
    confidences: NDArray[(Any,), np.float64],
    exact: bool = False,
    *args,
    **kwargs,
) -> Tuple[
    NDArray[(Any, 2, 2), np.float64],
    NDArray[(Any, 2, 2), np.float64],
    NDArray[(Any,), np.float64],
    NDArray[(Any,), np.float64],
    NDArray[(Any,), np.float64],
]:
    # For a given image (or set of images), compare each set of coordinates (np.ndarray of shape 4x2) to each
    # other set of coordinates. This is accomplished through a cartesian product of the coordinates list with
    # itself.
    cube_cartesian = itertools.product(cube_arr, cube_arr)
    confidences_cartesian = itertools.product(confidences, confidences)

    # We only want to calculate the iou for coordinate pairs that have not yet been evaluated. To accomplish that
    # we can add hashes of any already-checked coordinates to this set.
    checked_coordinates = set()

    # This output_shape is the number of unique pairs we should be evaluating.
    n = cube_arr.shape[0]
    output_shape = int((n ** 2 - n) / 2)

    # The below arrays are used to capture data from the below loop.
    arr1 = np.zeros((output_shape, cube_arr.shape[1], cube_arr.shape[2])).astype(
        np.float64
    )
    arr1_confidences = np.zeros(output_shape)
    arr2 = np.zeros((output_shape, cube_arr.shape[1], cube_arr.shape[2])).astype(
        np.float64
    )
    arr2_confidences = np.zeros(output_shape)

    i = 0
    # TODO: Replace with numpy-optimized routine.
    for box_arr, conf in zip(cube_cartesian, confidences_cartesian):
        if exact:
            if np.array_equal(box_arr[0], box_arr[1], *args, **kwargs):
                continue
        else:
            if np.allclose(box_arr[0], box_arr[1], *args, **kwargs):
                continue

        hash_prod0 = hash(box_arr[0].tobytes())
        hash_prod1 = hash(box_arr[1].tobytes())
        hash_prod = hash_prod0 * hash_prod1

        if hash_prod in checked_coordinates:
            continue
        checked_coordinates.add(hash_prod)

        arr1[i] = box_arr[0]
        arr1_confidences[i] = conf[0]
        arr2[i] = box_arr[1]
        arr2_confidences[i] = conf[1]
        i += 1

    ious = np.divide(cube.intersection(arr1, arr2), cube.union(arr1, arr2))
    return arr1, arr2, arr1_confidences, arr2_confidences, ious


def filter_used_boxes(filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2, selected_boxes_hash):
    # TODO: Optimize this loop.
    # Here, we're skipping evaluating any boxes that we've already selected.
    ixs_to_evaluate2 = []
    for j in np.arange(0, filtered_cube2.shape[0]):
        fc_hash = hash(filtered_cube2[j, :, :].tobytes())
        if fc_hash not in selected_boxes_hash:
            ixs_to_evaluate2.append(j)
        selected_boxes_hash.add(fc_hash)
    if filtered_cube1.shape[0] > 0:
        fc_hash = hash(filtered_cube1[0, :, :].tobytes())
        selected_boxes_hash.add(fc_hash)
        if fc_hash in selected_boxes_hash:
            filtered_cube1 = None
            filtered_confidences1 = None
    else:
        filtered_cube1 = None
        filtered_confidences1 = None
    filtered_cube2 = filtered_cube2[ixs_to_evaluate2, :, :]
    filtered_confidences2 = filtered_confidences2[ixs_to_evaluate2]

    return filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2, selected_boxes_hash


def evaluate_ious(
    ious: NDArray[(Any,), np.float64],
    cube1: NDArray[(Any, 2, 2), np.float64],
    confidences1: NDArray[(Any,), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    confidences2: NDArray[(Any,), np.float64],
    iou_threshold: float,
    selection_func: Callable[
        [
            NDArray[(Any, 2, 2), np.float64],
            NDArray[(Any, 2, 2), np.float64],
            NDArray[(Any,), np.float64],
            NDArray[(Any,), np.float64],
            Optional[Dict[str, Any]],
        ],
        Tuple[NDArray[(2, 2), np.float64], float],
    ],
    selection_kwargs: Dict[str, Any],
    round_decimals: int = 8,
) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
    cube1 = np.round(cube1, decimals=round_decimals)
    cube2 = np.round(cube2, decimals=round_decimals)

    selected_boxes = np.zeros((cube1.shape[0], 2, 2))
    selected_boxes_hash = set()
    selected_confidences = np.zeros(cube1.shape[0])

    unique_coords = np.unique(cube1, axis=0)
    for i in np.arange(0, unique_coords.shape[0]):
        # Evaluating coord1[i]
        this_unique_coord = unique_coords[i]

        # Get the indexes of coords1 that match this_unique_coord and that have an iou greater than or equal to
        # the threshold
        coords1_minus_this_unique_coord = cube1 - this_unique_coord
        coords1_different_from_this_unique_coord = np.sum(
            np.sum(coords1_minus_this_unique_coord, axis=2), axis=1
        )
        ixs_to_evaluate = np.argwhere(np.logical_and(
            (coords1_different_from_this_unique_coord == 0), (ious >= iou_threshold)
        ))


        # TODO: Need to filter out cube1/conf1 if it is in the selected_boxes_hash too.
        # Next, pull those indexes from the first dimension of the below arrays.
        filtered_cube1 = cube1[ixs_to_evaluate, :, :]  # Should all be the same.
        filtered_confidences1 = confidences1[ixs_to_evaluate]  # Should all be the same.
        filtered_cube2 = cube2[ixs_to_evaluate, :, :]  # Should be different values.
        filtered_confidences2 = confidences2[
            ixs_to_evaluate
        ]  # Should be different values.

        filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2, selected_boxes_hash = filter_used_boxes(filtered_cube1, filtered_confidences1, filtered_cube2, filtered_confidences2, selected_boxes_hash)

        if filtered_cube1 is not None:
            selected_box, selected_confidence = selection_func(
                filtered_cube1[:, 0, :, :],
                filtered_cube2[:, 0, :, :],
                filtered_confidences1.ravel(),
                filtered_confidences2.ravel(),
                selection_kwargs,
            )
        else:
            selected_box, selected_confidence = selection_func(
                filtered_cube1,
                filtered_cube2[:, 0, :, :],
                filtered_confidences1,
                filtered_confidences2.ravel(),
                selection_kwargs
            )

        if selected_box.shape[0] > 0:
            selected_boxes[i, :, :] = selected_box
            selected_confidences[i] = selected_confidence
    return selected_boxes[selected_confidences > 0, :, :], selected_confidences[selected_confidences > 0]
