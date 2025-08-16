#!/usr/bin/env python3
"""
Example entrypoint for FastLayer

This script demonstrates how to use the fastlayer framework.
It compares NumPy vs Numba optimized dot product results.
"""

import numpy as np
from fastlayer.core.hotpaths import dot_numpy, dot_numba


def main():
    size = 100000
    a = np.random.rand(size)
    b = np.random.rand(size)

    result_numpy = dot_numpy(a, b)
    result_numba = dot_numba(a, b)

    print("NumPy result :", result_numpy)
    print("Numba result:", result_numba)
    print("Difference   :", abs(result_numpy - result_numba))


if __name__ == "__main__":
    main()

