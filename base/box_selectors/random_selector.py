# Copyright Michael Ayabarreno-Hernandez. All rights reserved.

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple, Dict  # type: ignore


from base.box_selectors.base_selector import BaseSelector  # type: ignore


class RandomSelector(BaseSelector):
    def __init__(self):
        super().__init__()

    def select(
            self,
            box: NDArray[(Any, 2, 2), np.float64],
            cube: NDArray[(Any, 2, 2), np.float64],
            box_conf: NDArray[(Any,), np.float64],
            cube_confs: NDArray[(Any,), np.float64],
            kwawrgs: Dict[str, Any],
    ) -> Tuple[NDArray[(2, 2), np.float64], float]:
        """
        Args:
            box: Cube1 box. Can shape (0 or 1, 2, 2).
            cube: Cube2. Can be shape (Any, 2, 2).
            box_conf: Cube1 box confidence. Can be shape (0 or 1,).
            cube_confs: Cube2 box confidences. Can be shape (Any,).

        Returns:
            selected_box: Selected box from box and cube.
            selected_conf: Selected confidence from box_conf and cube_confs.
        """
        assert box is not None
        assert len(box.shape) == 2
        assert cube is not None
        assert box_conf is not None
        try:
            assert isinstance(box_conf, float)
        except Exception as e:
            print(box_conf)
            raise e
        assert cube_confs is not None

        print(cube.shape)
        cube = np.append(cube, np.expand_dims(box, 0), axis=0)
        cube_confs = np.append(cube_confs, box_conf)
        try:
            if cube.shape[0] == 1:
                return cube[0, :, :], cube_confs[0]
            else:
                ix = np.random.choice(np.arange(0, cube.shape[0] - 1))
        except Exception as e:
            print(f"CUBE_SHAPE: {cube.shape}")
            raise e

        print(cube.shape)
        print(cube_confs.shape)
        return cube[ix, :, :], cube_confs[ix]


def random_selector(
    cube1: NDArray[(Any, 2, 2), np.float64],
    cube2: NDArray[(Any, 2, 2), np.float64],
    confidences1: NDArray[(Any,), np.float64],
    confidences2: NDArray[(Any,), np.float64],
    kwargs
) -> Tuple[NDArray[(2, 2), np.float64], float]:
    """Chooses a random box from two cubes.

    Args:
        cube1: An array that implements the requirements of geometry.cube.assert_cube(). This represents a "stack" of
            bounding box coordinates. This values along 0th dimension of this array should all be identical. This
            represents the bounding box that we are comparing all other bounding boxes to in selection.
        cube2: An array that implements the requirements of geometry.cube.assert_cube(). This represents a "stack" of
            bounding box coordinates. This represents the bounding boxes that we are comparing to cube1.
        confidences1:
            An array that has a 0th dimension length equal to the 0th dimension length of cube1. This represents
            the confidence associated with each bounding box in cube1.
        confidences2:
            An array that has a 0th dimension length equal to the 0th dimension length of cube2. This represents
            the confidence associated with each bounding box in cube2.
        **kwargs:
            Unused kwargs to match expected function signature of nms/evaluate_ious/evaluate_ious().

    Returns:
        selected_box: An array that implements the requirements of geometry.box.assert_box(). This represents the
            selected bounding box.
        selected_confidence: This represents the selected confidence associated with the selected bounding box.

    Raises:
        AssertionError: This occurs if the 0th dimension of cube1 and confidences1 do not match or if the 0th
            dimension of cube2 and confidences2 do not match.
    """
    # assert cube1.shape[0] == confidences1.shape[0]
    # assert cube2.shape[0] == confidences2.shape[0]
    #
    if cube2.shape[0] == 0:
        return cube2, confidences2

    if cube1 is not None and cube1.shape[0] > 0:
        cube_combined = np.zeros((cube2.shape[0] + 1, 2, 2), dtype=np.float64)
        cube_combined[: cube2.shape[0], :, :] = cube2
        cube_combined[-1, :, :] = cube1[0, :, :]
        confidences_combined = np.zeros(confidences2.shape[0] + 1, dtype=np.float64)
        confidences_combined[: confidences2.shape[0]] = confidences2  # TODO: Check that this indexing is right.
        confidences_combined[-1] = confidences1[0]
    else:
        cube_combined = cube2
        confidences_combined = confidences2

    assert cube_combined.shape[0] > 0
    if cube_combined.shape[0] == 1:
        return cube_combined[0, :, :], confidences_combined[0]
    else:
        selected_ix = np.random.choice(np.arange(0, cube_combined.shape[0] - 1))  # TODO: Check that this indexing is right.
        return cube_combined[selected_ix, :, :], confidences_combined[selected_ix]
