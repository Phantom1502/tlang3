import pandas as pd
import numpy as np

from typing import List

from app.data_prepare.candle import Candle
from app.config.loader import load_config, get_scale, get_round_config
from app.config.schema import AppConfig, WindowConfig
from app.data_prepare.chartcodec import ChartCodec
from app.data_prepare.dataset_builder import DatasetBuilder

def window_high(charts: List[Candle]) -> float:
    return max(c.high for c in charts)

def window_low(charts: List[Candle]) -> float:
    return min(c.low for c in charts)

symbol = "XAUUSD"
timeframe = "M1"

cfg: AppConfig = load_config("configs")
scale = get_scale(cfg, symbol, timeframe)
window_cfg: WindowConfig = cfg.window

print(f"scale: {scale}, window_size: {window_cfg.window_size}")

df = pd.read_csv("data/preprocessed/train/XAUUSD_1Min.csv")
codec = ChartCodec(scale=scale, n_bins=cfg.base.n_bins)

window_ranges = []
for i in range(0, len(df) - window_cfg.window_size + 1):
    anchor_open = df.loc[i, "Open"]
    anchor_atr = df.loc[i, "ATR_100"]
    if anchor_atr <= 0 or np.isnan(anchor_atr):
        continue
    window = df.iloc[i:i + window_cfg.window_size]
    candles = codec.encode_window(window, anchor_open, anchor_atr)
    
    builder = DatasetBuilder(cfg, seed=0)
    sample = builder.build_pretrain_rows(candles, samples_per_chart=1)

    if sample is not None:
        print(sample)
        break