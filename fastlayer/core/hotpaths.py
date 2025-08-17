# core/hotpaths.py
from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
try:
    from numba import njit, prange, float64  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# runtime flags (health check may toggle these)
_DISABLE_CPP = False

log = logging.getLogger("fastlayer.hotpaths")
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(os.environ.get("FASTLAYER_LOGLEVEL", "WARNING"))

# -------- Public helpers --------

def prepare_arrays(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create two 1D float64 arrays of length n filled with random values."""
    a = np.random.rand(n).astype(np.float64, copy=False)
    b = np.random.rand(n).astype(np.float64, copy=False)
    return a, b

# -------- Baseline / hot paths --------

def dot_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product using NumPy (prefer BLAS path)."""
    return float(np.dot(a, b))

if _HAVE_NUMBA:
    @njit(float64(float64[:], float64[:]), nopython=True, cache=True, fastmath=True, parallel=True)
    def _dot_numba_impl(a, b):
        s = 0.0
        for i in prange(a.size):
            s += a[i] * b[i]
        return s

def dot_numba(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product using Numba njit (nopython, parallel)."""
    if not _HAVE_NUMBA:
        return dot_numpy(a, b)
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    return float(_dot_numba_impl(a64, b64))

# -------- Config & dispatch --------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default

def _cache_path() -> Path:
    return Path(os.environ.get("FASTLAYER_CACHE", str(Path.home() / ".cache"))) / "fastlayer.json"

def _load_cached_config() -> Dict[str, Any]:
    path = _cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def _save_cached_config(cfg: Dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        log.debug(f"could not save cache config: {e}")

def get_dispatch_config() -> dict:
    """Resolve dispatch configuration (defaults -> cache -> env)."""
    cfg = {"impl": "auto", "small": 8000, "med": 200000}
    # cache (autotune writes here)
    cached = _load_cached_config()
    for k in ("impl", "small", "med"):
        if k in cached:
            cfg[k] = cached[k]
    # environment overrides cache
    impl_env = _env_str("DOT_IMPL", "").lower()
    if impl_env:
        cfg["impl"] = impl_env
    small_env = os.environ.get("DOT_SMALL")
    med_env = os.environ.get("DOT_MED")
    if small_env and small_env.isdigit():
        cfg["small"] = int(small_env)
    if med_env and med_env.isdigit():
        cfg["med"] = int(med_env)
    return cfg

def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Auto-select an implementation based on problem size and availability.
    Environment overrides:
      DOT_IMPL=auto|numpy|numba|cpp
      DOT_SMALL=<int> (default 8000)
      DOT_MED=<int> (default 200000)
    """
    cfg = get_dispatch_config()
    impl = cfg["impl"]
    n = int(a.size)

    if impl != "auto":
        chosen = impl
    else:
        if n < cfg["small"]:
            chosen = "numpy"
        elif n < cfg["med"]:
            chosen = "numba"
        else:
            chosen = "cpp"

    if chosen == "numpy":
        log.debug(f"dot -> numpy (n={n})")
        return dot_numpy(a, b)

    if chosen == "numba":
        log.debug(f"dot -> numba (n={n})")
        return dot_numba(a, b)

    if chosen == "cpp":
        if _DISABLE_CPP:
            log.debug("cpp disabled by health check; falling back to numba")
            return dot_numba(a, b)
        try:
            import cpp_hot  # type: ignore
            a64 = np.ascontiguousarray(a, dtype=np.float64)
            b64 = np.ascontiguousarray(b, dtype=np.float64)
            log.debug(f"dot -> cpp (n={n})")
            return float(cpp_hot.dot_cpp(a64, b64))
        except Exception as e:
            log.debug(f"cpp not available ({e}); falling back to numba")
            return dot_numba(a, b)

    # Fallback
    log.debug(f"unknown DOT_IMPL={impl}, falling back to numba")
    return dot_numba(a, b)

# -------- Warmup --------

def warmup(sizes: Optional[Tuple[int, ...]] = (8192, 200_000), use_cpp: bool = True) -> None:
    """Pre-compile JITs and prime caches to amortize first-call cost."""
    a, b = prepare_arrays(sizes[0])
    _ = dot_numpy(a, b)
    _ = dot_numba(a, b)  # triggers Numba compilation

    if use_cpp:
        try:
            import cpp_hot  # type: ignore
            a2, b2 = prepare_arrays(sizes[-1])
            _ = cpp_hot.dot_cpp(a2.astype(np.float64), b2.astype(np.float64))
        except Exception:
            pass

    log.debug(f"warmup done for sizes={sizes} (cpp tried={use_cpp})")

# -------- Health check --------

def health_check() -> Dict[str, Any]:
    """Probe backends and mark unusable ones disabled."""
    global _DISABLE_CPP, _HAVE_NUMBA
    status = {"numpy": True, "numba": _HAVE_NUMBA, "cpp": False, "errors": {}}
    # numpy probe
    try:
        a,b = prepare_arrays(1024)
        _ = dot_numpy(a,b)
    except Exception as e:
        status["numpy"] = False
        status["errors"]["numpy"] = str(e)

    # numba probe
    if _HAVE_NUMBA:
        try:
            _ = dot_numba(a,b)
        except Exception as e:
            status["numba"] = False
            status["errors"]["numba"] = str(e)
            _HAVE_NUMBA = False

    # cpp probe
    try:
        import cpp_hot  # type: ignore
        _ = cpp_hot.dot_cpp(a,b)
        status["cpp"] = True
    except Exception as e:
        status["cpp"] = False
        status["errors"]["cpp"] = str(e)
        _DISABLE_CPP = True

    log.debug(f"health_check: {status}")
    return status

# -------- Autotuner --------

def _time_ms(fn, a, b, iters: int = 3) -> float:
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(a,b)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(ts) / len(ts))

def autotune(save: bool = True) -> Dict[str, int]:
    """Measure small grid to pick thresholds for this machine and cache them.
    Returns {"DOT_SMALL": v1, "DOT_MED": v2}.
    """
    # ensure compiled
    warmup()

    sizes = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
    small = 8000
    med = 200000

    # Find numpy -> numba crossover
    crossover_small = None
    for n in sizes:
        a,b = prepare_arrays(n)
        t_np = _time_ms(dot_numpy, a, b, iters=2)
        t_nb = _time_ms(dot_numba, a, b, iters=2)
        if t_nb < t_np:
            crossover_small = n
            break
    if crossover_small is not None:
        small = max(2000, crossover_small // 2)

    # Find numba -> cpp crossover (optional)
    crossover_med = None
    try:
        import cpp_hot  # type: ignore
        for n in sizes[::-1]:
            a,b = prepare_arrays(n)
            t_nb = _time_ms(dot_numba, a, b, iters=2)
            t_cpp = _time_ms(lambda x,y: float(cpp_hot.dot_cpp(x,y)), a, b, iters=2)
            if t_cpp < t_nb:
                crossover_med = n
                break
        if crossover_med is not None:
            med = max(50000, crossover_med)
    except Exception:
        pass

    cfg = {"DOT_SMALL": int(small), "DOT_MED": int(med)}
    if save:
        cache = _load_cached_config()
        cache.update({"small": cfg["DOT_SMALL"], "med": cfg["DOT_MED"]})
        _save_cached_config(cache)
    log.debug(f"autotune -> {cfg}")
    return cfg
