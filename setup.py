from setuptools import setup, find_packages, Extension
from pathlib import Path

def _np_include():
    try:
        import numpy as _np
        return _np.get_include()
    except Exception:
        return ""

def _pybind11_include():
    try:
        import pybind11
        return pybind11.get_include()
    except Exception:
        return ""

# Build extension as fastlayer.cpp_hot if source exists
cpp_src = Path(__file__).parent / "fastlayer" / "cpp_hot.cpp"
ext_modules = []
if cpp_src.exists():
    ext_modules = [
        Extension(
            "fastlayer.cpp_hot",
            sources=[str(cpp_src)],
            include_dirs=[_np_include(), _pybind11_include()],
            language="c++",
            extra_compile_args=["-O3"],
            optional=True,  # allow pure-Python install if build fails/missing
        )
    ]

setup(
    name="fastlayer",
    version="0.3.0",
    description="Soft-CPU Data Cache Engine",
    author="Karashi",
    license="Apache-2.0",
    packages=find_packages(include=["fastlayer*"]),
    python_requires=">=3.8",
    ext_modules=ext_modules,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
