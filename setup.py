from setuptools import setup, find_packages

setup(
    name="fastlayer",
    version="0.1.0",
    description="高速インメモリキャッシュ + NumPy/Numba最適化フレームワーク",
    author="Karashi",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

