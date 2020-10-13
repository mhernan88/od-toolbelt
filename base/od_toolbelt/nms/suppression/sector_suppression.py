# Copyright 2020 Michael Hernandez

import itertools  # type: ignore
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple, List  # type: ignore

from od_toolbelt.nms.metrics.base import Metric  # type: ignore
from od_toolbelt.nms.selection.base import Selector  # type: ignore
from od_toolbelt.nms.suppression.cartesian_product_suppression import CartesianProductSuppression  # type: ignore
from od_toolbelt import BoundingBoxArray, concatenate  # type: ignore


class SectorSuppression(CartesianProductSuppression):
    def __init__(
        self,
        metric: Metric,
        selector: Selector,
        sector_divisions: int,
    ):
        super().__init__(metric, selector)
        assert sector_divisions > 0
        self.sector_divisions = sector_divisions
        self.image_shape = (1, 1)

    def _create_sectors(
        self, sectors: List[NDArray[(2, 2), np.int64]], divide_on_height=True
    ) -> Tuple[List[NDArray[(2, 2), np.int64]], List[Tuple[int, bool]]]:
        """Divides an array into 2^n equal-sized sectors.

        Args:
            sectors: A list of sectors to divide. Normally, this will just include the image array coordinates as the
                only element.
            divide_on_height: Whether to begin with dividing the image array by height. If false, the first division
                will be by width (vertical).

        Returns:
            A similar list to the "sectors" argument. In this case, it will be the original array divided into 2^n
            subarrays, where n is self.sector_divisions.
        """
        divisions = 0
        dividing_lines = []
        while divisions < self.sector_divisions:
            new_sectors = []
            for sector in sectors:
                if divide_on_height:
                    dividing_line = (sector[0, 0] + sector[1, 0]) / 2
                    new_sector1 = np.array(
                        ((sector[0, 0], sector[0, 1]), (dividing_line, sector[1, 1])),
                        dtype=np.float64,
                    )
                    new_sector2 = np.array(
                        ((dividing_line, sector[0, 1]), (sector[1, 0], sector[1, 1])),
                        dtype=np.float64,
                    )
                else:
                    dividing_line = (sector[0, 1] + sector[1, 1]) / 2
                    new_sector1 = np.array(
                        ((sector[0, 0], sector[0, 1]), (sector[1, 0], dividing_line)),
                        dtype=np.float64,
                    )
                    new_sector2 = np.array(
                        ((sector[0, 0], dividing_line), (sector[1, 0], sector[1, 1])),
                        dtype=np.float64,
                    )
                new_sectors.append(new_sector1)
                new_sectors.append(new_sector2)
                dividing_lines.append((dividing_line, divide_on_height))
            sectors = new_sectors
            divisions += 1
        return sectors, dividing_lines

    def _handle_boundaries(
        self,
        bounding_box_array: BoundingBoxArray,
        dividing_lines: List[Tuple[int, bool]],
    ) -> BoundingBoxArray:
        """Finds all boxes overlapping boundaries. Performs selection on those boxes and any overlapping boxes.

        Args:
            bounding_box_array: The payload of the boxes, confidences, and labels.
            dividing_lines: The dividing_lines of each sector as determined by _create_sectors().

        Returns:
            selected_bids: The bids (box ids) selected by the selection algorithm.
            bounding_boxes: The original bounding boxes less any bounding boxes that were evaluated for selection.
            confidences: The original confidences less any confidences that were evaluated for selection.
        """
        on_boundary = np.full(
            bounding_box_array.bounding_boxes.shape[0], False, np.bool
        )
        all_bids = np.arange(0, bounding_box_array.bounding_boxes.shape[0])
        for bid in all_bids:
            for line, divide_on_height in dividing_lines:
                if (
                    divide_on_height
                    and bounding_box_array.bounding_boxes[bid, 0, 0]
                    < line
                    < bounding_box_array.bounding_boxes[bid, 1, 0]
                ):
                    on_boundary[bid] = True
                    break
                elif (
                    not divide_on_height
                    and bounding_box_array.bounding_boxes[bid, 0, 1]
                    < line
                    < bounding_box_array.bounding_boxes[bid, 1, 1]
                ):
                    on_boundary[bid] = True
                    break

        prod = itertools.product(all_bids[on_boundary], all_bids)
        selected_bids, evaluated_bids = self._evaluate_overlap(bounding_box_array, prod)
        evaluated_bids = np.asarray(list(evaluated_bids), dtype=np.int64)
        evaluated_bids_inv = np.ones(
            bounding_box_array.bounding_boxes.shape[0], np.bool
        )
        evaluated_bids_inv[evaluated_bids] = 0

        return bounding_box_array[np.asarray(selected_bids, dtype=np.int64)]

    @staticmethod
    def _in_sector(
        bounding_box: NDArray[(2, 2), np.float64], sector: NDArray[(2, 2), np.float64]
    ) -> bool:
        """Helper method to determine whether the center of a bounding box is in a given sector.

        Args:
            bounding_box: The bounding box you want to test.
            sector: The associated sector you want to test.

        Returns:
            A flag where True indicates that there is overlap.
        """
        bounding_box_center = np.array(
            (
                (bounding_box[0, 0] + bounding_box[1, 0]) / 2,
                (bounding_box[0, 1] + bounding_box[1, 1]) / 2,
            )
        )
        if (
            sector[0, 0] < bounding_box_center[0] < sector[1, 0]
            and sector[0, 1] < bounding_box_center[1] < sector[1, 1]
        ):
            return True
        return False

    def _assign_sectors(
        self,
        bounding_boxes: NDArray[(Any, 2, 2), np.int64],
        sectors: List[NDArray[(2, 2), np.float64]],
    ) -> List[List[int]]:
        """Assigns bounding boxes to sectors based on whether they overlap a sector.

        It is assumed that all bounding boxes that cross multiple sectors have already been processed and eliminated
        from the incoming dataset.

        Args:
            bounding_boxes: The bounding boxes you wish to assign.
            sectors: The sectors you wish to assign them to.

        Returns:
            A list of sectors. Each sector has a list of bounding boxes that it contains.
        """
        assigned_bids = set()
        all_sector_bids = []
        for sid in np.arange(0, len(sectors)):
            sector_bids = []
            for bid in np.arange(0, bounding_boxes.shape[0]):
                if bid in assigned_bids:
                    continue
                if self._in_sector(bounding_boxes[bid, :, :], sectors[sid]):
                    sector_bids.append(bid)
                    assigned_bids.add(bid)
            all_sector_bids.append(sector_bids)

        assert sum([len(x) for x in all_sector_bids]) == bounding_boxes.shape[0]
        return all_sector_bids

    def transform(
        self, bounding_box_array: BoundingBoxArray, *args, **kwargs
    ) -> BoundingBoxArray:
        """See base class documentation."""
        image_sector = [
            np.array(((0, 0), self.image_shape), dtype=np.int64),
        ]
        sectors, dividing_lines = self._create_sectors(image_sector)
        selected_bounding_box_arrays = [
            self._handle_boundaries(bounding_box_array, dividing_lines)
        ]

        all_sector_bids = self._assign_sectors(
            bounding_box_array.bounding_boxes, sectors
        )
        for sector_bids in all_sector_bids:
            assert isinstance(sector_bids, list)
            selected_bounding_box_arrays.append(
                self._cp_transform(
                    bounding_box_array[np.asarray(sector_bids, dtype=np.int64)]
                )
            )
        return concatenate(selected_bounding_box_arrays)
