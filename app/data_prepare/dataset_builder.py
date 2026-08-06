from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .candle import Candle, augment_shift
from app.config import (
    AppConfig,
    BaseConfig,
    WindowConfig
)
from app.data_prepare.generator import ZoneGenerator
from app.lang import (
    ASTVisitor,
)

class DatasetBuilder:
    def __init__(self, cfg: AppConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        base_cfg: BaseConfig = cfg.base
        window_cfg: WindowConfig = cfg.window
        self.input_candles = window_cfg.input_candles
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_bins = base_cfg.n_bins
        
        self.zone_gen = ZoneGenerator(cfg, seed=seed)
        self.ast_visitor = ASTVisitor(digit_pad=cfg.base.digit_pad)
        
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
        
        samples = self.zone_gen.generate_dataset(
            charts, 
            samples_per_chart=samples_per_chart
        )
        
        return [{"prompt": s.prompt, "completion": s.completion} for s in samples]
    
    def build_grpo_rows(
        self, 
        chart: List[Candle], 
        symbol: str, 
        index: int, 
        n_augments: int = 0
    ):
        rows: List[dict] = []

        variants: List[Tuple[str, List[Candle]]] = [(f"{symbol}_{index}", chart)]
        for k in range(n_augments):
            shifted = augment_shift(chart, self.rng, n_bins=self.n_bins)   # augment CẢ window (input+future)
            if shifted is not None:
                variants.append((f"{symbol}_{index}_aug{k}", shifted))

        for window_id, candles in variants:
            input_candles = candles[:self.input_candles]
            future_candles = candles[self.input_candles:]
            rows.append({
                "prompt": self.ast_visitor.render_chart_block(input_candles),
                "future_bins": [[c.open, c.high, c.low, c.close] for c in future_candles],
                "symbol": symbol,
                "window_id": window_id,
            })

        return rows