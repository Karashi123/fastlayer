# core/memdb.py
from collections import OrderedDict, defaultdict
import time
from typing import Any, Dict, Callable, Optional

class L1LRU:
    def __init__(self, capacity: int = 50_000):
        self.cap = capacity
        self.od: "OrderedDict[Any, tuple[Any, float, int]]" = OrderedDict()

    def get(self, k: Any) -> Optional[Any]:
        if k in self.od:
            v, ts, h = self.od.pop(k)
            self.od[k] = (v, ts, h+1)
            return v
        return None

    def put(self, k: Any, v: Any, hits: int = 0) -> None:
        if k in self.od:
            self.od.pop(k)
        elif len(self.od) >= self.cap:
            self.od.popitem(last=False)
        self.od[k] = (v, time.time(), hits)

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

    def get(self, k: Any) -> Optional[Any]:
        v = self.l1.get(k)
        if v is not None:
            return v
        v = self.l2.get(k)
        if v is not None and self.l2.hits[k] >= self.promote_hits:
            self.l1.put(k, v, hits=self.l2.hits[k])
        return v

    def put(self, k: Any, v: Any, write_through: bool = False) -> None:
        self.l2.data[k] = v
        if write_through:
            self.l1.put(k, v)
        else:
            self.l2.hits[k] = max(self.l2.hits[k], 1)
