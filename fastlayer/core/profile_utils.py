# core/profile_utils.py
import cProfile, pstats, io
from contextlib import contextmanager

@contextmanager
def profile(label: str = "profile"):
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(30)
        print(f"\n=== {label} ===\n{s.getvalue()}")
