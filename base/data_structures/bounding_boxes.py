import numpy as np
from warnings import warn
from nptyping import NDArray
from typing import Any, Iterator, Optional, Union


class BoundingBoxArray:
    dtypes = {
        np.int8: "np.int8",
        np.int16: "np.int16",
        np.int32: "np.int32",
        np.int64: "np.int64",
        np.float32: "np.float32",
        np.float64: "np.float64"
    }

    def __init__(
            self,
            bounding_boxes: NDArray[(Any, 2, 2), np.float64],
            confidences: NDArray[(Any,), np.float64],
            labels: NDArray[(Any,), np.int64]
    ):
        self.bounding_boxes = bounding_boxes
        self.confidences = confidences
        self.labels = labels

    def __iter__(self):
        for i in np.arange(0, self.bounding_boxes.shape[0]):
            yield i, self.bounding_boxes[i, :, :], self.confidences[i], self.labels[i]

    def __len__(self):
        return self.bounding_boxes.shape[0]

    def read_csv(self, filename):
        # TODO: Read CSV with columns xmin, ymin, xmax, ymax, confidence, label
        pass

    def write_csv(self, filename):
        # TODO: Write CSV with same cols as read_csv().
        pass

    def _check_numpy_warning(self, variable_label: str, options: Union[Iterator[str], Iterator[np.dtype]]):
        text = f"In BoundingBoxesArray.check(), {variable_label} dtype is not "
        options = [opt if isinstance(opt, str) else self.dtypes[opt] for opt in options]

        if len(options) == 1:
            warn(f"{text} {options[0]}")
        else:
            options_string = ", ".join(options)
            warn(f"{text} one of: {options_string}", SyntaxWarning)

    def append(
            self,
            bounding_box: Union[NDArray[(2, 2), np.float64], NDArray[(1, 2, 2), np.float64]],
            confidence: Union[float, NDArray[(1,), np.float64]],
            label: Union[int, NDArray[(1,), np.int64]],
    ):
        if len(bounding_box.shape) == 2:
            bounding_box = np.expand_dims(bounding_box, 0)
        self.bounding_boxes = np.append(self.bounding_boxes, bounding_box, axis=0)

        if isinstance(confidence, float):
            confidence = np.array((confidence,), dtype=np.float64)
        self.confidences = np.append(self.confidences, confidence, axis=0)

        if isinstance(label, int):
            label = np.array((label,), dtype=np.int64)
        self.labels = np.append(self.labels, label, axis=0)

    def check(
            self,
            bounding_box: Optional[NDArray[(2, 2), np.float64]] = None,
            confidence: Optional[float] = None,
            label: Optional[int] = None,
    ):
        valid_floats = (np.float32, np.float64)
        valid_ints = (np.int8, np.int16, np.int32, np.int64)

        if bounding_box is not None:
            assert isinstance(bounding_box, np.ndarray)
            assert np.all(bounding_box >= 0)
            assert np.all(bounding_box <= 1)
            assert len(bounding_box.shape) == 2
            assert bounding_box.shape[0] == 2 and bounding_box.shape[0] == 2
            if bounding_box.dtype not in valid_floats:
                self._check_numpy_warning("bounding_box", valid_floats)
                warn("In BoundingBoxArray.check(), bounding_box dtype is not np.float64")
        if confidence is not None:
            if not isinstance(confidence, float):
                warn("In BoundingBoxArray.check(), confidence is not float")
        if label is not None:
            if not isinstance(label, int):
                warn("In BoundingBoxArray.check(), label is not int")

        assert isinstance(self.bounding_boxes, np.ndarray)
        assert np.all(self.bounding_boxes >= 0)
        assert np.all(self.bounding_boxes <= 1)
        assert len(self.bounding_boxes.shape) == 3
        assert self.bounding_boxes.shape[1] == 2 and self.bounding_boxes.shape[2] == 2
        if self.bounding_boxes.dtype not in valid_floats:
            self._check_numpy_warning("self.bounding_boxes", valid_floats)
        assert isinstance(self.confidences, np.ndarray)
        assert len(self.confidences.shape) == 1
        if self.confidences.dtype not in valid_floats:
            self._check_numpy_warning("self.confidences", valid_floats)
        assert isinstance(self.labels, np.ndarray)
        assert len(self.labels.shape) == 1
        if self.labels.dtype not in valid_ints:
            self._check_numpy_warning("self.labels", valid_ints)
        assert self.bounding_boxes.shape[0] == self.confidences.shape[0]
        assert self.bounding_boxes.shape[0] == self.labels.shape[0]
