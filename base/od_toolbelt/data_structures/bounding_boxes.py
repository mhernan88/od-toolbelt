# Copyright 2020 Michael Hernandez

import csv
import json
import numpy as np
from warnings import warn
from nptyping import NDArray
from typing import Any, Iterator, Optional, Union, Dict, List, Tuple


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

        assert bounding_boxes.shape[0] == confidences.shape[0]
        assert bounding_boxes.shape[0] == labels.shape[0]

        self.bounding_boxes = bounding_boxes
        self.confidences = confidences
        self.labels = labels

        if bounding_box_ids is None:
            self.bounding_box_ids = np.arange(0, bounding_boxes.shape[0])
        else:
            self.bounding_box_ids = bounding_box_ids

    def __getitem__(
            self,
            item: NDArray[(Any,), np.int64]
    ):
        if item.shape[0] == 0:
            return None
        # TODO: Optimize loop over self.bounding_box_ids
        ixs = [True if bid in item else False for bid in self.bounding_box_ids]

        new_bounding_box_array = BoundingBoxArray(
            bounding_boxes=self.bounding_boxes[ixs, :, :],
            confidences=self.confidences[ixs],
            labels=self.labels[ixs],
            bounding_box_ids=self.bounding_box_ids[ixs],
        )
        new_bounding_box_array.check()

        return new_bounding_box_array

    def __iter__(
            self
    ):
        for i in np.arange(0, self.bounding_boxes.shape[0]):
            yield i, self.bounding_boxes[i, :, :], self.confidences[i], self.labels[i]

    def __len__(
            self
    ):
        return self.bounding_boxes.shape[0]

    def __str__(
            self
    ):
        ids = ", ".join([str(x) for x in self.bounding_box_ids.tolist()])
        return f"A bounding box array with ids: {ids}"

    def _check_values_and_update(
            self,
            bounding_boxes: NDArray[(Any, 2, 2), np.float64],
            confidences: NDArray[(Any,), np.float64],
            labels: NDArray[(Any,), np.int64],
            bounding_box_ids: NDArray[(Any,), np.int64],
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

    def to_dict(
            self
    ) -> List[
        Dict[str, Union[float, int, Dict[str, float]]]
    ]:
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

    def from_json(
            self,
            filename: str,
            ignore_incoming_tag_ids: bool
    ):
        with open(filename, "r") as f:
            payload = json.load(f)
        self.from_dict(payload, ignore_incoming_tag_ids)

    def to_json(
            self,
            filename: str,
    ):
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

    def bounding_box_id_to_ix(
            self,
            bid: int
    ):
        return np.argwhere(self.bounding_box_ids == int(bid))

    def lookup_box(
            self,
            bid: int
    ):
        box = self.bounding_boxes[self.bounding_box_ids == int(bid), :, :][0]
        assert len(box.shape) == 2
        return box

    def _check_numpy_warning(
            self,
            variable_label: str,
            options: Union[Iterator[str], Iterator[np.dtype]]
    ):
        text = f"In BoundingBoxesArray.check(), {variable_label} dtype is not "
        options = [opt if isinstance(opt, str) else self.dtypes[opt] for opt in options]

        if len(options) == 1:
            warn(f"{text} {options[0]}")
        else:
            options_string = ", ".join(options)
            warn(f"{text} one of: {options_string}", SyntaxWarning)

    def check_lengths(self):
        prefix = f"In BoundingBoxesArray.check_lengths, self.bounding_boxes (len={self.bounding_boxes.shape[0]}) had different length than"
        try:
            assert self.bounding_boxes.shape[0] == self.confidences.shape[0]
        except AssertionError:
            raise ValueError(f"{prefix} self.confidences (len={self.confidences.shape[0]})")

        try:
            assert self.bounding_boxes.shape[0] == self.labels.shape[0]
        except AssertionError:
            raise ValueError(f"{prefix} self.labels (len={self.labels.shape[0]})")

        try:
            assert self.bounding_boxes.shape[0] == self.bounding_box_ids.shape[0]
        except AssertionError:
            raise ValueError(f"{prefix} self.bounding_box_ids (len={self.bounding_box_ids.shape[0]})")

    def append(
        self,
        bounding_box: Union[
            NDArray[(2, 2), np.float64],
            NDArray[(1, 2, 2), np.float64],
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
        self.bounding_box_ids = np.append(
            self.bounding_box_ids, bounding_box_id, axis=0
        )
        self.check_lengths()

    def _check_new(
            self,
            valid_floats: Tuple,
            bounding_box: Optional[NDArray[(2, 2), np.float64]] = None,
            confidence: Optional[float] = None,
            label: Optional[int] = None,
    ):
        prefix = "In BoundingBoxArray._check_new(),"
        if bounding_box is not None:
            try:
                assert isinstance(bounding_box, np.ndarray)
            except AssertionError:
                raise ValueError(
                    f"{prefix} bounding_box was the wrong type: "
                    f"expected=np.ndarray, actual={type(bounding_box)}"
                )

            try:
                assert np.all(bounding_box >= 0)
            except AssertionError:
                raise ValueError(
                    f"{prefix} bounding_box had values < 0"
                )

            try:
                assert np.all(bounding_box <= 1)
            except AssertionError:
                raise ValueError(
                    f"{prefix} bounding_box had values > 1"
                )

            try:
                assert len(bounding_box.shape) == 2
            except AssertionError:
                raise ValueError(
                    f"{prefix} bounding_box had wrong number of dimensions: "
                    f"expected=2, actual={len(bounding_box.shape)}"
                )
            assert bounding_box.shape[0] == 2 and bounding_box.shape[0] == 2
            if bounding_box.dtype not in valid_floats:
                self._check_numpy_warning("bounding_box", valid_floats)
                warn(
                    f"{prefix} bounding_box had wrong dtype: "
                    f"expected={', '.join([self.dtypes[x] for x in valid_floats])}, actual={bounding_box.dtype}"
                )

        if confidence is not None:
            if not isinstance(confidence, float):
                warn(
                    f"{prefix} confidence had the wrong dtype: expected=float, actual={type(confidence)}"
                )

        if label is not None:
            if not isinstance(label, int):
                warn(
                    f"{prefix} label had teh wrong dtype: expected=int, actual={type(label)}"
                )

    def _check_existing(self, valid_floats: Tuple, valid_ints: Tuple):
        prefix = "In BoundingBoxArray._check_existing(),"
        try:
            assert isinstance(self.bounding_boxes, np.ndarray)
        except AssertionError:
            raise ValueError(f"{prefix} self.bounding_boxes was the wrong type: "
                             f"expected=np.ndarray, actual={type(self.bounding_boxes)}")

        try:
            assert np.all(self.bounding_boxes >= 0)
        except AssertionError:
            raise ValueError(f"{prefix} self.bounding_boxes had values < 0")

        try:
            assert np.all(self.bounding_boxes <= 1)
        except AssertionError:
            raise ValueError(f"{prefix} self.bounding_boxes had values > 1")

        try:
            assert len(self.bounding_boxes.shape) == 3
        except AssertionError:
            raise ValueError(f"{prefix} self.bounding_boxes had wrong number of dimensions: "
                             f"expected=3, actual={len(self.bounding_boxes.shape)}")

        try:
            assert self.bounding_boxes.shape[1] == 2
        except AssertionError:
            raise ValueError(f"{prefix} self.bounding_boxes had wrong shape for dimension 1 (starting at 0): "
                             f"expected=2, actual={self.bounding_boxes.shape[1]}")

        try:
            assert self.bounding_boxes.shape[2] == 2
        except AssertionError:
            raise ValueError(f"{prefix} self.bounding_boxes had wrong shape for dimension 2 (starting at 0): "
                             f"expected=2, actual={self.bounding_boxes.shape[2]}")

        if self.bounding_boxes.dtype not in valid_floats:
            self._check_numpy_warning("self.bounding_boxes", valid_floats)

        try:
            assert isinstance(self.confidences, np.ndarray)
        except AssertionError:
            raise ValueError(f"{prefix} self.confidences had wrong type: "
                             f"expected=np.ndarray, actual={type(self.confidences)}")

        try:
            assert len(self.confidences.shape) == 1
        except AssertionError:
            raise ValueError(f"{prefix} self.confidences had wrong number of dimensions: "
                             f"expected=1, actual={len(self.confidences.shape)}")

        if self.confidences.dtype not in valid_floats:
            self._check_numpy_warning("self.confidences", valid_floats)

        try:
            assert isinstance(self.labels, np.ndarray)
        except AssertionError:
            raise ValueError(f"{prefix} self.labels had wrong type: "
                             f"expected=np.ndarray, actual={type(self.labels)}")

        try:
            assert len(self.labels.shape) == 1
        except AssertionError:
            raise ValueError(f"{prefix} self.labels had wrong number of dimensions: "
                             f"expected=1, actual={len(self.labels.shape)}")

        if self.labels.dtype not in valid_ints:
            self._check_numpy_warning("self.labels", valid_ints)

    def check(
        self,
        bounding_box: Optional[NDArray[(2, 2), np.float64]] = None,
        confidence: Optional[float] = None,
        label: Optional[int] = None,
    ):
        valid_floats = (np.float32, np.float64)
        valid_ints = (np.int8, np.int16, np.int32, np.int64)

        self._check_new(valid_floats, bounding_box, confidence, label)
        self._check_existing(valid_floats, valid_ints)
        self.check_lengths()


def concatenate(bounding_box_arrays: List[BoundingBoxArray]):
    """Concatenates multiple bounding box arrays into a single one.

    Args:
        bounding_box_arrays:

    Returns:

    """
    bounding_box_arrays = [bb for bb in bounding_box_arrays if bb is not None]

    # TODO: Remove tty except wrapper
    try:
        assert len(bounding_box_arrays) > 0
    except AssertionError:
        ValueError("In concatenate(), bounding_box_arrays had length of 0 (greater than 0 was expected)")
    output = bounding_box_arrays[0]
    for ix, bounding_box_array in enumerate(bounding_box_arrays):
        if ix == 0:
            output = bounding_box_array
        else:
            old_bounding_boxes = output.bounding_boxes
            old_confidences = output.confidences
            old_labels = output.labels
            old_bounding_box_ids = output.bounding_box_ids

            new_bounding_boxes = bounding_box_array.bounding_boxes
            new_confidences = bounding_box_array.confidences
            new_labels = bounding_box_array.labels
            new_bounding_box_ids = bounding_box_array.bounding_box_ids

            output = BoundingBoxArray(
                bounding_boxes=np.concatenate((old_bounding_boxes, new_bounding_boxes), axis=0),
                confidences=np.concatenate((old_confidences, new_confidences), axis=0),
                labels=np.concatenate((old_labels, new_labels), axis=0),
                bounding_box_ids=np.concatenate((old_bounding_box_ids, new_bounding_box_ids), axis=0),
            )
    return output
