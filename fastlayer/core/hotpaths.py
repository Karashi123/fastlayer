from __future__ import annotations

import os, json, time, logging, importlib, threading
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Callable
from tempfile import NamedTemporaryFile

import numpy as np
try:
    from numba import njit, prange, float64 as nb_f64, float32 as nb_f32  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# runtime flags (health check may toggle these) — writes are lock-guarded
_DISABLE_CPP = False
_LAST_IMPL: str = "none"
_state_lock = threading.RLock()

log = logging.getLogger("fastlayer.hotpaths")
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(os.environ.get("FASTLAYER_LOGLEVEL", "WARNING"))

def _set_disable_cpp(v: bool) -> None:
    global _DISABLE_CPP
    with _state_lock:
        _DISABLE_CPP = v

def _set_last_impl(v: str) -> None:
    global _LAST_IMPL
    with _state_lock:
        _LAST_IMPL = v

# -------- Public helpers --------

_MAX_AUTOTUNE_N = 1 << 24  # DoS防止の上限

def prepare_arrays(n: int, dtype: np.dtype = np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """Create two 1D arrays of length n filled with random values."""
    if n < 1 or n > _MAX_AUTOTUNE_N:
        raise ValueError(f"n out of bounds: {n}")
    a = np.random.rand(n).astype(dtype, copy=False)
    b = np.random.rand(n).astype(dtype, copy=False)
    return a, b

# -------- Baseline / hot paths --------

def dot_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product using NumPy (prefer BLAS path)."""
    return float(np.dot(a, b))

if _HAVE_NUMBA:
    @njit(nb_f64(nb_f64[:], nb_f64[:]), nopython=True, cache=True, fastmath=True, parallel=True)
    def _dot_numba_f64(a, b):
        s = 0.0
        for i in prange(a.size):
            s += a[i] * b[i]
        return s

    @njit(nb_f32(nb_f32[:], nb_f32[:]), nopython=True, cache=True, fastmath=True, parallel=True)
    def _dot_numba_f32(a, b):
        s = nb_f32(0.0)
        for i in prange(a.size):
            s += a[i] * b[i]
        return s

def dot_numba(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product using Numba njit (nopython, parallel)."""
    if not _HAVE_NUMBA:
        return dot_numpy(a, b)
    if a.dtype == np.float32 and b.dtype == np.float32:
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        b32 = np.ascontiguousarray(b, dtype=np.float32)
        return float(_dot_numba_f32(a32, b32))
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    return float(_dot_numba_f64(a64, b64))

# -------- Config & dispatch --------

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default

def _cache_path() -> Path:
    base = Path(os.environ.get("FASTLAYER_CACHE", str(Path.home() / ".cache")))
    try:
        base = base.expanduser().resolve()
    except Exception:
        base = Path.home().joinpath(".cache")
    return base / "fastlayer.json"

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
        with NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
            json.dump(cfg, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)  # atomic save
    except Exception as e:
        log.debug(f"could not save cache config: {e}")

def get_dispatch_config() -> dict:
    """Resolve dispatch configuration (defaults -> cache -> env)."""
    cfg: Dict[str, Any] = {"impl": "auto", "small": 8000, "med": 200_000}
    cache = _load_cached_config()
    if "small" in cache:
        cfg["small"] = int(cache["small"])
    if "med" in cache:
        cfg["med"] = int(cache["med"])
    impl_env = _env_str("DOT_IMPL", cfg["impl"])
    if impl_env:
        cfg["impl"] = impl_env
    small_env = os.environ.get("DOT_SMALL")
    if small_env:
        cfg["small"] = int(small_env)
    med_env = os.environ.get("DOT_MED")
    if med_env:
        cfg["med"] = int(med_env)
    # legacy names kept for compatibility
    cfg["DOT_SMALL"] = cfg["small"]
    cfg["DOT_MED"] = cfg["med"]
    return cfg

def _validate_1d_same_len(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"dot expects 1D arrays, got a.ndim={a.ndim}, b.ndim={b.ndim}")
    if a.size != b.size:
        raise ValueError(f"size mismatch: a.size={a.size}, b.size={b.size}")
    return a, b

def _validate_finite(a: np.ndarray, b: np.ndarray) -> None:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("inputs contain NaN/Inf")

def get_last_backend() -> str:
    return _LAST_IMPL

def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Auto-select an implementation based on problem size and availability.
    Environment overrides:
      DOT_IMPL=auto|numpy|numba|cpp
      DOT_SMALL=<int> (default 8000)
      DOT_MED=<int> (default 200000)
    """
    global _DISABLE_CPP

    _validate_1d_same_len(a, b)
    _validate_finite(a, b)
    cfg = get_dispatch_config()
    impl = cfg["impl"]
    n = int(a.size)

    if impl != "auto":
        chosen = impl
    else:
        if n < int(cfg["small"]):
            chosen = "numpy"
        elif n < int(cfg["med"]):
            chosen = "numba"
        else:
            chosen = "cpp"

    if chosen == "numpy":
        log.debug(f"dot -> numpy (n={n})")
        _set_last_impl("numpy"); return dot_numpy(a, b)

    if chosen == "numba":
        log.debug(f"dot -> numba (n={n})")
        _set_last_impl("numba"); return dot_numba(a, b)

    if chosen == "cpp":
        if _DISABLE_CPP:
            log.debug("cpp disabled; falling back to numba")
            _set_last_impl("numba(fallback_cpp_disabled)")
            return dot_numba(a, b)
        try:
            # secure import inside package namespace (fallback to legacy ext name)
            try:
                cpp_hot = importlib.import_module("fastlayer.cpp_hot")
            except Exception:
                cpp_hot = importlib.import_module("cpp_hot")
            a64 = np.ascontiguousarray(a, dtype=np.float64)
            b64 = np.ascontiguousarray(b, dtype=np.float64)
            log.debug(f"dot -> cpp (n={n})")
            _set_last_impl("cpp"); return float(cpp_hot.dot_cpp(a64, b64))
        except Exception as e:
            log.debug(f"cpp not available ({e}); falling back to numba")
            _set_disable_cpp(True)
            _set_last_impl("numba(fallback_cpp_import_error)")
            return dot_numba(a, b)

    # Fallback
    log.debug(f"unknown DOT_IMPL={impl}, falling back to numba")
    _set_last_impl("numba(fallback_unknown_impl)"); return dot_numba(a, b)

# -------- Warmup --------

def warmup(sizes: Optional[Tuple[int, ...]] = (8192, 200_000), use_cpp: bool = True) -> None:
    """Pre-compile JITs and prime caches to amortize first-call cost."""
    a, b = prepare_arrays(sizes[0], dtype=np.float64)
    _ = dot_numpy(a, b)
    _ = dot_numba(a, b)  # Numba f64
    a32, b32 = prepare_arrays(sizes[0], dtype=np.float32)
    _ = dot_numba(a32, b32)  # Numba f32

    if use_cpp and not _DISABLE_CPP:
        try:
            try:
                cpp_hot = importlib.import_module("fastlayer.cpp_hot")
            except Exception:
                cpp_hot = importlib.import_module("cpp_hot")
            a2, b2 = prepare_arrays(sizes[-1], dtype=np.float64)
            _ = cpp_hot.dot_cpp(a2, b2)
        except Exception:
            _set_disable_cpp(True)

    log.debug(f"warmup done for sizes={sizes} (cpp tried={use_cpp})")

# -------- Health check --------

def health_check() -> Dict[str, Any]:
    """Probe backends and mark unusable ones disabled."""
    global _HAVE_NUMBA
    status = {"numpy": True, "numba": _HAVE_NUMBA, "cpp": False, "errors": {}}
    a,b = prepare_arrays(1024, dtype=np.float64)
    try:
        _ = dot_numpy(a,b)
    except Exception as e:
        status["numpy"] = False
        status["errors"]["numpy"] = str(e)

    if _HAVE_NUMBA:
        try:
            _ = dot_numba(a,b)
        except Exception as e:
            status["numba"] = False
            status["errors"]["numba"] = str(e)
            with _state_lock:
                _HAVE_NUMBA = False

    # cpp probe
    try:
        try:
            cpp_hot = importlib.import_module("fastlayer.cpp_hot")
        except Exception:
            cpp_hot = importlib.import_module("cpp_hot")
        if not hasattr(cpp_hot, "dot_cpp"):
            raise AttributeError("fastlayer.cpp_hot.dot_cpp missing")
        _ = cpp_hot.dot_cpp(a,b)
        status["cpp"] = True
    except Exception as e:
        status["cpp"] = False
        status["errors"]["cpp"] = str(e)
        _set_disable_cpp(True)

    log.debug(f"health_check: {status}")
    return status

# -------- Autotuner --------

def _median_ms(fn: Callable[[np.ndarray, np.ndarray], float], a: np.ndarray, b: np.ndarray, iters: int = 3) -> float:
    ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        _ = fn(a, b)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    mid = len(ts) // 2
    return float(ts[mid]) if ts else 0.0

def _warn_threads():
    for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","OMP_NUM_THREADS"):
        if not os.environ.get(k):
            log.debug(f"hint: consider setting {k} to avoid oversubscription")

def autotune(save: bool = True, iters_small: int = 3, iters_large: int = 3) -> Dict[str, int]:
    """Measure small grid to pick thresholds for this machine and cache them.
    Returns {"DOT_SMALL": v1, "DOT_MED": v2}.
    """
    _warn_threads()
    warmup()

    sizes = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
    small = 8000
    med = 200_000

    # Find numpy -> numba crossover
    crossover_small = None
    for n in sizes:
        a,b = prepare_arrays(n)
        t_np = _median_ms(dot_numpy, a, b, iters=iters_small)
        t_nb = _median_ms(dot_numba, a, b, iters=iters_small)
        if t_nb < t_np:
            crossover_small = n
            break
    if crossover_small is not None:
        small = max(2000, crossover_small // 2)

    # Find numba -> cpp crossover (optional)
    crossover_med = None
    try:
        try:
            cpp_hot = importlib.import_module("fastlayer.cpp_hot")
        except Exception:
            cpp_hot = importlib.import_module("cpp_hot")
        for n in sizes[::-1]:
            a,b = prepare_arrays(n)
            t_nb = _median_ms(dot_numba, a, b, iters=iters_large)
            t_cpp = _median_ms(lambda x,y: float(cpp_hot.dot_cpp(x,y)), a, b, iters=iters_large)
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
