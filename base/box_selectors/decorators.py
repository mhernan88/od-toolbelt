from logging import Logger
from functools import wraps

import numpy as np
from nptyping import NDArray
from typing import Any, Dict


def assert_select_arguments(function):
    def wrapper(*args, **kwargs):
        try:
            assert box is not None
        except Exception as e:
            print("box argument to selector was None")
            raise e

        try:
            assert len(box.shape) == 2
        except Exception as e:
            print("box argument to selector to shape had invalid shape")
            raise e

        try:
            assert cube is not None
        except Exception as e:
            print("cube argument to selector was None")
            raise e

        try:
            assert box_conf is not None
        except Exception as e:
            print("box_conf argument to selector was None")
            raise e

        try:
            assert isinstance(box_conf, float)
        except Exception as e:
            print("box_conf argument to selector was not float")
            raise e

        try:
            assert cube_confs is not None
        except Exception as e:
            print("cube_confs argument to selector was not None")
            raise e

        function(box, cube, box_conf, cube_confs, kwargs)
    return wrapper