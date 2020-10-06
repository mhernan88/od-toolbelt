import csv
import json
import numpy as np
from warnings import warn
from nptyping import NDArray
from typing import Any, Iterator, Optional, Union, Dict, List


class BoundingBoxArray:
    dtypes = {
        np.int8: "np.int8",
        np.int16: "np.int16",
        np.int32: "np.int32",
        np.int64: "np.int64",
        np.float32: "np.float32",
        np.float64: "np.float64",
    }

    def __init__(
        self,
        bounding_boxes: NDArray[(Any, 2, 2), np.float64],
        confidences: NDArray[(Any,), np.float64],
        labels: NDArray[(Any,), np.int64],
        bounding_box_ids: Optional[NDArray[(Any,), np.int64]] = None,
    ):
        if isinstance(bounding_boxes, tuple):
            bounding_boxes = np.asarray(bounding_boxes, dtype=np.float64)
        if isinstance(confidences, tuple):
            confidences = np.asarray(confidences, dtype=np.float64)
        if isinstance(labels, tuple):
            labels = np.asarray(labels, dtype=np.int64)
        if isinstance(bounding_box_ids, tuple):
            bounding_box_ids = np.asarray(bounding_box_ids, dtype=np.int64)

        self.bounding_boxes = bounding_boxes
        self.confidences = confidences
        self.labels = labels

        if bounding_box_ids is None:
            self.bounding_box_ids = np.arange(0, bounding_boxes.shape[0])
        else:
            self.bounding_box_ids = bounding_box_ids

    def __iter__(self):
        for i in np.arange(0, self.bounding_boxes.shape[0]):
            yield i, self.bounding_boxes[i, :, :], self.confidences[i], self.labels[i]

    def __len__(self):
        return self.bounding_boxes.shape[0]

    def _check_values_and_update(
        self, bounding_boxes, confidences, labels, bounding_box_ids
    ):
        bounding_boxes = np.asarray(bounding_boxes, dtype=np.float64)
        confidences = np.asarray(confidences, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        bounding_box_ids = np.asarray(bounding_box_ids, dtype=np.int64)

        assert bounding_boxes.shape[0] == confidences.shape[0]
        assert bounding_boxes.shape[0] == labels.shape[0]
        assert bounding_boxes.shape[0] == bounding_box_ids.shape[0]

        self.bounding_boxes = bounding_boxes
        self.confidences = confidences
        self.labels = labels
        self.bounding_box_ids = bounding_box_ids

    def to_dict(self) -> List[Dict[str, Union[float, int, Dict[str, float]]]]:
        bounding_boxes = []
        for i in np.arange(0, self.bounding_boxes.shape[0]):
            label = self.labels[i]

            bounding_boxes.append(
                {
                    "probability": self.confidences[i],
                    "tagName": label,
                    "boundingBox": {
                        "left": self.bounding_boxes[i, 0, 1],
                        "top": self.bounding_boxes[i, 0, 0],
                        "width": np.subtract(
                            self.bounding_boxes[i, 1, 1], self.bounding_boxes[i, 0, 1]
                        ),
                        "height": np.subtract(
                            self.bounding_boxes[i, 1, 0], self.bounding_boxes[i, 0, 0]
                        ),
                    },
                    "tagId": self.bounding_box_ids[i],
                }
            )
        return bounding_boxes

    def from_dict(
        self,
        payload: List[Dict[str, Union[float, int, str, Dict[str, float]]]],
        ignore_incoming_tag_ids: bool,
        append: bool = False,
    ):
        if not append:
            # If we're not appending, then we're overwriting and should start with new lists.
            bounding_boxes, confidences, labels, bounding_box_ids = [], [], [], []
        else:
            # Otherwise, load the existing data.
            bounding_boxes = self.bounding_boxes.tolist()
            confidences = self.confidences.tolist()
            labels = self.labels.tolist()
            bounding_box_ids = self.bounding_box_ids.tolist()

        tag_id = 0  # Default tag id in case it's not provided.
        evaluated_tag_ids = set(
            bounding_box_ids
        )  # Guarantees that incoming tag_ids are unique.

        for item in payload:
            bounding_boxes.append(
                (
                    (item["boundingBox"]["top"], item["boundingBox"]["left"]),
                    (
                        item["boundingBox"]["top"] + item["boundingBox"]["height"],
                        item["boundingBox"]["left"] + item["boundingBox"]["width"],
                    ),
                )
            )
            confidences.append(item["probability"])

            # In case the user provides strings in the payload, we need to convert them to integers.
            if isinstance(item["tagName"], str):
                labels.append(item["tagName"])
            else:
                labels.append(item["tagName"])

            # If a tag_id is provided, then get it; Otherwise use the default.
            if "tagId" in item.keys() and not ignore_incoming_tag_ids:
                this_tag_id = item["tagId"]
            else:
                this_tag_id = tag_id
            if this_tag_id in evaluated_tag_ids:
                raise ValueError("All tagIds in payload must be unique")
            bounding_box_ids.append(this_tag_id)
            evaluated_tag_ids.add(this_tag_id)
            tag_id += 1
        self._check_values_and_update(
            bounding_boxes, confidences, labels, bounding_box_ids
        )

    def from_json(self, filename, ignore_incoming_tag_ids: bool):
        with open(filename, "r") as f:
            payload = json.load(f)
        self.from_dict(payload, ignore_incoming_tag_ids)

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    def from_csv(self, filename, append: bool = False):
        if not append:
            # If we're not appending, then we're overwriting and should start with new lists.
            bounding_boxes, confidences, labels, bounding_box_ids = [], [], [], []
        else:
            # Otherwise, load the existing data.
            bounding_boxes = self.bounding_boxes.tolist()
            confidences = self.confidences.tolist()
            labels = self.labels.tolist()
            bounding_box_ids = self.bounding_box_ids.tolist()

        with open(filename, newline="") as f:
            reader = csv.reader(f, delimeter=",", quotechar='"')
            for row in reader:
                bounding_box_ids.append(row[0])
                labels.append(row[1])
                confidences.append(row[2])
                new_bbox = ((row[3], row[4]), (row[5], row[6]))
                bounding_boxes.append(new_bbox)

        self._check_values_and_update(
            bounding_boxes, confidences, labels, bounding_box_ids
        )

    def to_csv(self, filename):
        rows = [("tagId", "tagName", "probability", "top", "left", "height", "width")]

        for i in np.arange(0, self.bounding_boxes.shape[0]):
            row = (
                self.bounding_box_ids[i],
                self.labels[i],
                self.confidences[i],
                self.bounding_boxes[i, 0, 0],
                self.bounding_boxes[i, 0, 1],
                np.subtract(self.bounding_boxes[i, 1, 1], self.bounding_boxes[i, 0, 1]),
                np.subtract(self.bounding_boxes[i, 1, 0], self.bounding_boxes[i, 0, 0]),
            )
            rows.append(row)

        with open(filename, mode="w") as f:
            writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerows(rows)

    def _check_numpy_warning(
        self, variable_label: str, options: Union[Iterator[str], Iterator[np.dtype]]
    ):
        text = f"In BoundingBoxesArray.check(), {variable_label} dtype is not "
        options = [opt if isinstance(opt, str) else self.dtypes[opt] for opt in options]

        if len(options) == 1:
            warn(f"{text} {options[0]}")
        else:
            options_string = ", ".join(options)
            warn(f"{text} one of: {options_string}", SyntaxWarning)

    def _check_lengths(self):
        assert self.bounding_boxes.shape[0] == self.confidences.shape[0]
        assert self.bounding_boxes.shape[0] == self.labels.shape[0]
        assert self.bounding_boxes.shape[0] == self.bounding_box_ids.shape[0]

    def append(
        self,
        bounding_box: Union[
            NDArray[(2, 2), np.float64], NDArray[(1, 2, 2), np.float64]
        ],
        confidence: Union[float, NDArray[(1,), np.float64]],
        label: Union[int, NDArray[(1,), np.int64]],
        bounding_box_id: Optional[Union[int, NDArray[(1,), np.int64]]] = None,
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

        if bounding_box_id is not None:
            if isinstance(bounding_box_id, int):
                bounding_box_id = np.array((bounding_box_id,), dtype=np.int64)
        else:
            bounding_box_id = np.array((np.max(self.bounding_box_ids) + 1,), dtype=np.int64)
        print(f"BBID SHAPE: {bounding_box_id.shape}")
        print(f"SELF.BBID SHAPE: {self.bounding_box_ids.shape[0]}")
        self.bounding_box_ids = np.append(
            self.bounding_box_ids, bounding_box_id, axis=0
        )
        self._check_lengths()

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
                warn(
                    "In BoundingBoxArray.check(), bounding_box dtype is not np.float64"
                )
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
        self._check_lengths()
