from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .candle import Candle
from app.config.schema import (
    AppConfig,
    BaseConfig,
    WindowConfig
)
from app.data_prepare.generator import ZoneGenerator

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

class DatasetBuilder:
    def __init__(self, cfg: AppConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        base_cfg: BaseConfig = cfg.base
        window_cfg: WindowConfig = cfg.window
        self.input_candles = window_cfg.input_candles
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_bins = base_cfg.n_bins
        
    def build_pretrain_rows(
        self,
        chart: List[Candle],
        samples_per_chart: int = 4,
        n_augments: int = 0,
    ):
        candles_inputs: List[Candle] = chart[:self.input_candles]
        charts: List[List[Candle]] = [candles_inputs]
        for _ in range(n_augments):
            shifted = augment_shift(candles_inputs, self.rng, n_bins=self.n_bins)
            if shifted is not None:
                charts.append(shifted)
                
        zone_gen : ZoneGenerator = ZoneGenerator(cfg=self.cfg, seed=self.seed)
        
        samples = zone_gen.generate_dataset(
            charts, 
            samples_per_chart=samples_per_chart
        )
        
        return [{"prompt": s.prompt, "completion": s.completion} for s in samples]