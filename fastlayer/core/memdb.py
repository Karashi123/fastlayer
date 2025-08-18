from collections import OrderedDict, defaultdict
import time, threading
from typing import Any, Dict, Optional

class L1LRU:
    def __init__(self, capacity: int = 50_000):
        self.cap = capacity
        self.od: "OrderedDict[Any, tuple[Any, float, int]]" = OrderedDict()

    def get(self, k: Any) -> Optional[Any]:
        if k in self.od:
            v, ts, h = self.od.pop(k)
            self.od[k] = (v, time.monotonic(), h + 1)  # refresh ts & hit++
            return v
        return None

    def put(self, k: Any, v: Any, hits: Optional[int] = None) -> None:
        if k in self.od:
            _, _, old_h = self.od.pop(k)
            h = old_h if hits is None else hits
        elif len(self.od) >= self.cap:
            self.od.popitem(last=False)
            h = 0 if hits is None else hits
        else:
            h = 0 if hits is None else hits
        self.od[k] = (v, time.monotonic(), h)

    def __len__(self) -> int:
        return len(self.od)

class L2Store:
    def __init__(self, data: Dict[Any, Any]):
        self.data = data
        self.hits = defaultdict(int)

    def get(self, k: Any) -> Optional[Any]:
        v = self.data.get(k)
        if v is not None:
            self.hits[k] += 1
        return v

class MemDB:
    def __init__(self, l2_data: Dict[Any, Any], l1_cap: int = 50_000, promote_hits: int = 2):
        self.l1 = L1LRU(l1_cap)
        self.l2 = L2Store(l2_data)
        self.promote_hits = promote_hits
        self.l1_hits = 0
        self.l2_hits = 0
        self._lock = threading.RLock()

    def get(self, k: Any) -> Optional[Any]:
        with self._lock:
            v = self.l1.get(k)
            if v is not None:
                self.l1_hits += 1
                return v
            v = self.l2.get(k)
            if v is not None:
                self.l2_hits += 1
                if self.l2.hits[k] >= self.promote_hits:
                    self.l1.put(k, v, hits=self.l2.hits[k])
            return v

    def put(self, k: Any, v: Any, write_through: bool = False) -> None:
        with self._lock:
            self.l2.data[k] = v
            if write_through:
                self.l1.put(k, v)
            else:
                self.l2.hits[k] = max(self.l2.hits[k], 1)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.l1_hits + self.l2_hits
            return {
                "l1_hits": self.l1_hits,
                "l2_hits": self.l2_hits,
                "hit_ratio": (self.l1_hits / total) if total else 0.0,
                "l1_size": len(self.l1),
                "l2_size": len(self.l2.data),
                "promote_hits": self.promote_hits,
            }

    def delete(self, k: Any) -> None:
        with self._lock:
            self.l1.od.pop(k, None)
            self.l2.data.pop(k, None)
            self.l2.hits.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self.l1.od.clear(); self.l2.data.clear(); self.l2.hits.clear()
            self.l1_hits = self.l2_hits = 0
