import numpy as np # type: ignore
from pathlib import Path # type: ignore
from setuptools import Extension, setup  # type: ignore
from Cython.Build import cythonize # type: ignore

rootdir = Path(__file__).parent.absolute()

extensions = [
    Extension(
        "od_toolbelt.nms.metrics.base",
        [
            Path(rootdir, "src", "base.pyx").as_posix()
        ],
        include_dirs=[np.get_include()]
    ),
]

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
    ext_modules=cythonize(extensions),
)
