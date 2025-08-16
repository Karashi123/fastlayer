from setuptools import setup, find_packages

setup(
    name="fastlayer",
    version="0.1.0",
    description="Soft-CPU Data Cache Engine",
    author="Karashi",
    license="Apache-2.0",
    packages=find_packages(include=["fastlayer*"]),  # ← パッケージを限定
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

