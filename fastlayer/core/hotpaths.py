# core/hotpaths.py
import numpy as np
from typing import Tuple

# NumPy�i����C/Fortran�j�Ńx�N�g����
def dot_numpy(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum())

# Numba��JIT�iJIT������Python���[�v���@�B��ցj
try:
    from numba import njit
    _NUMBA = True
except Exception:
    _NUMBA = False

if _NUMBA:
    @njit(cache=True, fastmath=True)
    def dot_numba(a: np.ndarray, b: np.ndarray) -> float:
        s = 0.0
        for i in range(a.size):
            s += a[i] * b[i]
        return s
else:
    def dot_numba(a: np.ndarray, b: np.ndarray) -> float:
        # �t�H�[���o�b�N
        return dot_numpy(a, b)

def prepare_arrays(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random(n, dtype=np.float64), rng.random(n, dtype=np.float64)
