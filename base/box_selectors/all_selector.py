import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple, Dict  # type: ignore


class AllSelector:
    def __init__(self):
        super().__init__()

    def select(
            self,
            box: NDArray[(Any, 2, 2), np.float64],
            cube: NDArray[(Any, 2, 2), np.float64],
            box_conf: NDArray[(Any,), np.float64],
            cube_confs: NDArray[(Any,), np.float64],
            kwargs: Dict[str, Any]
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
        assert isinstance(box_conf, float)
        assert cube_confs is not None

        cubes_combined = np.append(cube, box)
        confs_combined = np.append(cube_confs, box_conf)

        return cubes_combined, confs_combined
