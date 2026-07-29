from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CandleNode:
    o: int
    h: int
    l: int
    c: int


@dataclass
class ChartNode:
    candles: List[CandleNode] = field(default_factory=list)


@dataclass
class ZoneNode:
    direction: str        # "support" | "resistance"
    lower_bin: int
    upper_bin: int


@dataclass
class ThinkNode:
    trend: Optional[str] = None                  # "UP" | "DOWN" | "RANGE"
    current_price_bin: Optional[int] = None      # BẮT BUỘC theo spec — luôn phải có mặt
    zone: Optional[ZoneNode] = None

@dataclass
class ProgramNode:
    chart: Optional[ChartNode] = None
    think: Optional[ThinkNode] = None