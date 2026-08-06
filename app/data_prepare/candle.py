from dataclasses import dataclass
from typing import List, Optional
import random

@dataclass
class Candle:
    open: int
    high: int
    low: int
    close: int
    
    @property
    def o(self) -> int: 
        return self.open
    
    @property
    def h(self) -> int: 
        return self.high
    
    @property
    def l(self) -> int: 
        return self.low
    
    @property
    def c(self) -> int: 
        return self.close

def augment_shift(
    candles: List[Candle],
    rng: random.Random,
    n_bins: int,
) -> Optional[List[Candle]]:
    """Trả None nếu window đã chiếm hết biên độ bin (không còn chỗ dịch)."""
    lows = [c.low for c in candles]
    highs = [c.high for c in candles]
    min_low, max_high = min(lows), max(highs)

    shift_min = -min_low
    shift_max = (n_bins - 1) - max_high
    if shift_min > shift_max:
        return None

    choices = [d for d in range(shift_min, shift_max + 1) if d != 0]
    if not choices:
        return None

    delta = rng.choice(choices)
    return [Candle(c.open + delta, c.high + delta, c.low + delta, c.close + delta) for c in candles]