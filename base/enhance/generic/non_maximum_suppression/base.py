from typing import List, Optional


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

    def transform(self, predictions: List[list], *args, **kwargs) -> list:
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
