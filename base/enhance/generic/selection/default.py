import numpy as np


class DefaultSelectionAlgorithm:
    def __init__(self, coordinates1, coordinates2, confidences1, confidences2):
        self.coordinates1 = coordinates1
        self.coordinates2 = coordinates2
        self.confidences1 = confidences1
        self.confidences2 = confidences2

    def dispatch(self, method):
        if method.lower() == "first":
            return self._select_first()
        elif method.lower() == "last":
            return self._select_last()
        elif method.lower() == "random":
            return self._select_random()
        elif method.lower() == "confidence":
            return self._select_confidence()
        else:
            raise ValueError(
                "Invalid value provided for method argument | actual={method}, expected='first' "
                "| 'last' | 'random' | 'confidence'"
            )

    def _select_first(self):
        return self.coordinates1[0], self.confidences1[0]

    def _select_last(self):
        return self.coordinates2[-1], self.confidences2[-1]

    def _get_all_coords(self):
        all_coords = np.zeros(
            self.coordinates2.shape[0] + 1,
            self.coordinates2.shape[1],
            self.coordinates2.shape[2],
        )
        all_coords[0, :, :] = self.coordinates1[0, :, :]
        all_coords[1:, :, :] = self.coordinates2
        return all_coords

    def _get_all_confidences(self):
        all_confidences = np.zeros(self.confidences2.shape[0] + 1)
        all_confidences[0] = self.confidences1[0]
        all_confidences[1:] = self.confidences2
        return all_confidences

    def _select_random(self):
        all_coords = self._get_all_coords()
        all_confidences = self._get_all_confidences()

        selected_ix = np.random.choice(enumerate(all_coords))[0]
        selected_coord = all_coords[selected_ix, :, :]
        selected_confidence = all_confidences[selected_ix]
        return selected_coord, selected_confidence

    def _select_confidence(self):
        all_coords = self._get_all_coords()
        all_confidences = self._get_all_confidences()

        selected_ix = np.argmax(all_confidences)
        selected_coord = all_coords[selected_ix, :, :]
        selected_confidence = all_confidences[selected_ix]
        return selected_coord, selected_confidence
