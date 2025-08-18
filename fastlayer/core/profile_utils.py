import cProfile, pstats, io, os, sys
from contextlib import contextmanager
from typing import Optional, Literal, IO
from pathlib import Path
from tempfile import NamedTemporaryFile

SortKey = Literal["cumulative", "tottime", "calls", "ncalls", "time"]

def _dump_stats_safe(ps: pstats.Stats, out_path: str) -> None:
    p = Path(out_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=str(p.parent)) as tmp:
        ps.dump_stats(tmp.name)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(p)

@contextmanager
def profile(
    label: str = "profile",
    sort: SortKey = "cumulative",
    limit: int = 30,
    stream: Optional[IO[str]] = None,
    out_path: Optional[str] = None,
    strip_dirs: bool = True,
    enabled: Optional[bool] = None,
):
    """
    CPUプロファイルを取得し要約を出力、必要なら .prof を安全に保存。
    - 有効化: enabled=True または環境変数 FL_PROFILE=1/true/yes/on
    - sort: cumulative|tottime|calls|ncalls|time
    - limit: 表示件数
    - stream: 出力先（デフォルト stderr）
    - out_path: pstatsダンプ（原子的保存）
    - strip_dirs: パス短縮
    """
    if enabled is None:
        val = os.getenv("FL_PROFILE", "")
        enabled = val.lower() in ("1", "true", "yes", "on")

    if not enabled:
        # プロファイル無効時は素通り
        yield
        return

    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        out = stream if stream is not None else sys.stderr
        ps = pstats.Stats(pr, stream=out)
        if strip_dirs:
            ps.strip_dirs()
        key = {
            "cumulative": pstats.SortKey.CUMULATIVE,
            "tottime": pstats.SortKey.TIME,
            "time": pstats.SortKey.TIME,
            "calls": pstats.SortKey.CALLS,
            "ncalls": pstats.SortKey.CALLS,
        }.get(sort, pstats.SortKey.CUMULATIVE)
        ps.sort_stats(key)
        print(f"\n=== {label} (sort={sort}, limit={limit}) ===", file=out)
        ps.print_stats(limit)
        if out_path:
            try:
                _dump_stats_safe(ps, out_path)
                print(f"[profile] dumped stats -> {out_path}", file=out)
            except Exception as e:
                print(f"[profile] dump failed: {e}", file=out)
