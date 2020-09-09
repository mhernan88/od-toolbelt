from typing import List  # type: ignore


class Selector:
    """
    A selector is how one of multiple overlapping images is selected.
    """

    def __init__(self, *args, **kwargs):
        """Any configuration variables can be passed and stored here.
        """
        pass

    def select(self, bids: List[int]) -> int:
        """Selects one box id of many.

        Args:
            bids: The box ids you wish to consider.

        Returns:
            A single box id from the bids argument.
        """
        raise NotImplementedError
