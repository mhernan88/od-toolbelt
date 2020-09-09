import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, List  # type: ignore


class Suppressor:
    """
    A base suppressor class to serve as a specification for other suppressor classes.
    """

    def __init__(self):
        pass

    def transform(
        self, bounding_boxes: NDArray[(Any, 2, 2), np.float64], *args, **kwargs
    ) -> List[int]:
        """A method that filters down the amount of bounding boxes of an image

        Args:
            bounding_boxes: The bounding boxes that have been identified on the image. The 0th dimension of the array
            represents the amount of bounding boxes. The 1st dimension of the array represents the top-left and
            bottom-right points of the bounding box. The 2nd dimension of the array represents the y and x coordinates
            of each point.

        Returns:
            A list of indexes of which boxes to keep.
        """
        raise NotImplementedError

    def burst(
        self, bounding_box_burst: NDArray[(Any, Any, 2, 2), np.float64], *args, **kwargs
    ) -> List[int]:
        """A method to run transform() on a burst of images.

        All images in the burst should be take of the same subject from the same angle.

        Args:
            bounding_box_burst: A burst of images. The 0th dimension of the array represents the number of photos in
            the burst. All subsequent dimensions are the same as transform().

        Returns:
            A list of indexes of which boxes to keep.
        """
        raise NotImplementedError

    def batch(
        self,
        bounding_box_batch: NDArray[(Any, Any, Any, 2, 2), np.float64],
        *args,
        **kwargs
    ) -> List[List[int]]:
        """A method to run burst() on a batch of different images.

        Args:
            bounding_box_batch: A batch of bursts of images. The 0th dimension fo the array represents the number of
            groups of images to process. All subsequent dimensions are the same as burst().

        Returns:
            A list. Each element of the list is an output of burst().
        """
        raise NotImplementedError
