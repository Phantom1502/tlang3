from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from app.candle import Candle

CandleNode = Candle

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
    
    @property
    def zone_type(self) -> Optional[str]:
        if self.zone is None:
            return "NO_ZONE"
        return "SUP_ZONE" if self.zone.direction == "support" else "RES_ZONE"
    
    @property
    def zone_upper(self) -> int:
        if self.zone is None:
            return 0
        return self.zone.upper_bin
    
    @property
    def zone_lower(self) -> int:
        if self.zone is None:
            return 0
        return self.zone.lower_bin

@dataclass
class ProgramNode:
    chart: Optional[ChartNode] = None
    think: Optional[ThinkNode] = None