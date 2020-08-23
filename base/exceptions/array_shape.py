class WrongNumberOfDimensionsError(Exception):
    def __init__(self, message):
        super(WrongNumberOfDimensionsError, self).__init__(message)


class WrongDimensionShapeError(Exception):
    def __init__(self, message):
        super(WrongDimensionShapeError, self).__init__(message)


class MismatchedFirstDimensionError(Exception):
    def __init__(self, message):
        super(MismatchedFirstDimensionError, self).__init__(message)
