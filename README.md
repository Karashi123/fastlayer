# FastLayer

[![GitHub release](https://img.shields.io/github/v/release/Karashi123/fastlayer)](https://github.com/Karashi123/fastlayer/releases)
[![CI](https://github.com/Karashi123/fastlayer/actions/workflows/ci.yml/badge.svg)](https://github.com/Karashi123/fastlayer/actions)


**FastLayer is a Soft-CPU Data Cache Engine.**  
A new kind of runtime framework for Python, inspired by CPU memory hierarchy.

**FastLayer はソフトウェア実装の CPU データキャッシュエンジンです。**  
CPU のメモリ階層構造に着想を得た、新しいタイプの Python ランタイムフレームワークです。

## Installation

### From AUR (Arch Linux / Manjaro)
```bash
yay -S python-fastlayer-git

### Example
from fastlayer import MemDB, dot, warmup, autotune, health_check, get_last_backend
import numpy as np

print("health:", health_check())
warmup()
print("tuned:", autotune())

X = np.arange(1000, dtype=np.float64)
Y = np.arange(1000, dtype=np.float64)
print("dot:", dot(X, Y), "backend:", get_last_backend())

db = MemDB(l2_data={"X": X, "Y": Y})
print("L2→L1 warm:", db.get("X") is not None)
print("stats:", db.stats())

### サンプル (JP)
from fastlayer import MemDB, dot, warmup, autotune, health_check, get_last_backend
import numpy as np

print("ヘルス:", health_check())
warmup()
print("チューニング:", autotune())

X = np.arange(1000, dtype=np.float64)
Y = np.arange(1000, dtype=np.float64)
print("内積:", dot(X, Y), "実装:", get_last_backend())

db = MemDB(l2_data={"X": X, "Y": Y})
print("ウォーム:", db.get("X") is not None)
print("統計:", db.stats())

