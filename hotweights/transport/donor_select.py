from __future__ import annotations

from typing import Dict, List, Tuple


def _ip_prefix(addr: str, bits: int = 24) -> str:
    parts = addr.split(".")
    if len(parts) != 4:
        return addr
    if bits == 24:
        return ".".join(parts[:3])
    if bits == 16:
        return ".".join(parts[:2])
    return addr


def order_donors(workers: Dict[str, dict], ranks: List[int], self_meta: dict) -> List[Tuple[str, dict]]:
    """Order donors by topology closeness.

    Preference order:
    - same host
    - same /24 prefix
    - same /16 prefix
    - others
    Returns list of (worker_id, meta) sorted by preference.
    """
    self_host = self_meta.get("host")
    self_addr = self_meta.get("addr")
    pref: List[Tuple[int, str, dict]] = []
    for wid, meta in workers.items():
        if int(meta.get("rank", -1)) not in ranks:
            continue
        score = 3
        if self_host and meta.get("host") == self_host:
            score = 0
        elif self_addr and meta.get("addr") and _ip_prefix(meta["addr"]) == _ip_prefix(self_addr):
            score = 1
        elif self_addr and meta.get("addr") and _ip_prefix(meta["addr"], 16) == _ip_prefix(self_addr, 16):
            score = 2
        pref.append((score, wid, meta))
    pref.sort(key=lambda x: (x[0], x[1]))
    return [(wid, meta) for score, wid, meta in pref]

