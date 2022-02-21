import numpy as np # type: ignore
from pathlib import Path # type: ignore
from setuptools import Extension, setup  # type: ignore

rootdir = Path(__file__).parent.absolute()

setup(
    name="odToolbelt",
    version="0.1.1",
    include_package_data=True,
    packages=[
        "od_toolbelt",
        "od_toolbelt.data_structures",
        "od_toolbelt.nms",
        "od_toolbelt.nms.metrics",
        "od_toolbelt.nms.selection",
        "od_toolbelt.nms.suppression",
        "od_toolbelt.nms.aggregators",
    ],
    url="",
    license="",
    author="Michael Hernandez",
    author_email="",
    description="",
)
