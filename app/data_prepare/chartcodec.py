import re
import numpy as np
import pandas as pd
from typing import List
from .candle import Candle

_TOKEN_RE = re.compile(r"([OHLC])_(\d+)")

class ChartCodec:
    def __init__(self, scale: float, n_bins: int):
        self.scale = scale
        self.n_bins = n_bins
        
    def quantize_price(self, price, anchor_open, anchor_atr) -> int:
        if anchor_atr <= 0 or np.isnan(anchor_atr):
            raise ValueError("anchor_atr phải > 0")
        
        norm = (price - anchor_open) / (self.scale * anchor_atr)
        norm = np.clip(norm, -1.0, 1.0)
        bin_idx = int(round((norm + 1.0) / 2.0 * (self.n_bins - 1)))
        return bin_idx
    
    def dequantize_bin(self, bin_idx, anchor_open, anchor_atr) -> float:
        norm = (bin_idx / (self.n_bins - 1)) * 2.0 - 1.0
        price = anchor_open + norm * self.scale * anchor_atr
        return price
    
    def encode_window(self, window_df: pd.DataFrame, anchor_atr) -> List[Candle]:
        max_high = window_df['High'].max()
        min_low = window_df['Low'].min()
        
        # 2. Xác định tâm đối xứng của window
        anchor_open = (max_high + min_low) / 2.0
        
        candles = []
        for _, row in window_df.iterrows():
            o = self.quantize_price(row['Open'], anchor_open, anchor_atr)
            h = self.quantize_price(row['High'], anchor_open, anchor_atr)
            l = self.quantize_price(row['Low'], anchor_open, anchor_atr)
            c = self.quantize_price(row['Close'], anchor_open, anchor_atr)
            candles.append(Candle(o, h, l, c))
            
        return candles, anchor_open
    
    def decode_window(self, text: str, anchor_open, anchor_atr) -> str:
        buckets = {"O": [], "H": [], "L": [], "C": []}
        for letter, num in _TOKEN_RE.findall(text):
            buckets[letter].append(int(num))
        
        n_candles = len(buckets["O"])
        if not all(len(buckets[k]) == n_candles for k in "HLC"):
            raise ValueError(
                f"Số token O/H/L/C không khớp nhau: "
                f"O={len(buckets['O'])} H={len(buckets['H'])} "
                f"L={len(buckets['L'])} C={len(buckets['C'])} "
                f"— text có thể bị model sinh lỗi/thiếu token."
            )
        
        rows = []
        for i in range(n_candles):
            rows.append({
                "Open":  self.dequantize_bin(buckets["O"][i], anchor_open, anchor_atr),
                "High":  self.dequantize_bin(buckets["H"][i], anchor_open, anchor_atr),
                "Low":   self.dequantize_bin(buckets["L"][i], anchor_open, anchor_atr),
                "Close": self.dequantize_bin(buckets["C"][i], anchor_open, anchor_atr),
            })
        
        return " ".join([f"O_{row['Open']} H_{row['High']} L_{row['Low']} C_{row['Close']}" for row in rows])