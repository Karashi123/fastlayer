# FastLayer

**FastLayer is a Soft-CPU Data Cache Engine.**  
A new kind of runtime framework for Python, inspired by CPU memory hierarchy.

**FastLayer はソフトウェア実装の CPU データキャッシュエンジンです。**  
CPU のメモリ階層構造に着想を得た、新しいタイプの Python ランタイムフレームワークです。

## Features (v0.2.0)

- **Warmup**: preload hot data into memory before execution
- **Autotune**: profile and select the best implementation (Python, NumPy, Cython, C++)
- **Health Check**: monitor cache hit ratio and memory usage
- **Automatic Dispatch**: run optimized kernels (`dot()`, custom ops) transparently
- **Logging & Debugging**: visualize which backend is used

## 機能 (v0.2.0)

- **ウォームアップ (Warmup)**: 実行前にホットデータをメモリへ事前展開
- **自動チューニング (Autotune)**: 実行時に最適な実装（Python/NumPy/Cython/C++）を選択
- **ヘルスチェック (Health Check)**: キャッシュヒット率やメモリ使用量を監視
- **自動ディスパッチ (Automatic Dispatch)**: `dot()` など最適化カーネルを透過的に利用
- **ロギング & デバッグ**: どのバックエンドが使われているか可視化

## Installation

### From AUR (Arch Linux / Manjaro)
```bash
yay -S python-fastlayer-git


---

### 4. サンプルコード
```markdown
## Example

```python
from fastlayer import memDB, hotpaths
import numpy as np

# Create DB
db = memDB()

# Warmup hot data
db.warmup({"X": [1,2,3], "Y": [4,5,6]})

# Autotune dot product
X = np.arange(1000, dtype=np.float32)
Y = np.arange(1000, dtype=np.float32)
res = hotpaths.autotune("dot", X, Y)
print("Dot result:", res)

# Health check
print(db.health_check())

###

from fastlayer import memDB, hotpaths
import numpy as np

db = memDB()
db.warmup({"X": [1,2,3], "Y": [4,5,6]})

X = np.arange(1000, dtype=np.float32)
Y = np.arange(1000, dtype=np.float32)
res = hotpaths.autotune("dot", X, Y)
print("内積:", res)

print(db.health_check())  # ヘルスチェック結果を表示

