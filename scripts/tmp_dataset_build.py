from __future__ import annotations

import random
from typing import List, Optional, Tuple, Literal
import numpy as np

from app.training.reward import TLangReward
from tlang import (
    ChartCodec,
    CandleNode,
    ChartNode,
    ThinkNode,
    ZoneNode,
    ProgramNode,
    TLangConfig,
    ASTVisitor
)

from app.config import AppConfig, load_config

def find_truly_valid_zones(
    future_candles: List[CandleNode],
    mode: Literal["support", "resistance"] = "support",
    swing_window: int = 2,
    zone_width: int = 50,
    noise: int = 10,
    max_bin: int = 2047
) -> List[Tuple[int, int, int]]:
    """
    Tìm các Zone (Support hoặc Resistance) KHẢ DỤNG THỰC SỰ tính từ t=0.
    
    :param future_candles: Danh sách nến tương lai [CandleNode]
    :param mode: "support" hoặc "resistance"
    :param swing_window: Số nến bên trái và bên phải để xác nhận Swing High/Low (mặc định = 2)
    :param zone_width: Độ rộng của Zone (mặc định = 50 bins)
    :param max_bin: Giới hạn Bin tối đa (mặc định = 2047)
    :return: List các Tuple (future_idx, lower_bin, upper_bin)
    """
    valid_zones = []
    n = len(future_candles)
    
    if n == 0:
        return valid_zones

    is_support = (mode == "support")

    for i in range(n):
        # 1. KIỂM TRA ĐIỀU KIỆN SWING HIGH / SWING LOW
        # Đảm bảo đủ số nến swing_window ở hai bên
        left_start = max(0, i - swing_window)
        right_end = min(n - 1, i + swing_window)

        if is_support:
            current_val = future_candles[i].low
            # Là Swing Low nếu giá Low hiện tại <= tất cả các nến trong cửa sổ xung quanh
            is_swing = all(current_val <= future_candles[j].low for j in range(left_start, right_end + 1) if j != i)
        else:
            current_val = future_candles[i].high
            # Là Swing High nếu giá High hiện tại >= tất cả các nến trong cửa sổ xung quanh
            is_swing = all(current_val >= future_candles[j].high for j in range(left_start, right_end + 1) if j != i)

        if not is_swing:
            continue

        noise_value = random.randint(0,noise)
        # 2. XÁC ĐỊNH LOWER_BIN VÀ UPPER_BIN
        if is_support:
            lower_bin = max(0, current_val - zone_width - noise_value)
            upper_bin = lower_bin + zone_width + noise_value
        else:
            upper_bin = min(max_bin, current_val + zone_width + noise_value)
            lower_bin = upper_bin - zone_width - noise_value

        # 3. PRISTINE CHECK (Kiểm tra tính Khả dụng từ t=0 đến i-1)
        # Bỏ qua các Zone đã bị chạm/xuyên qua bởi các nến đứng trước
        is_available = True
        for prev_idx in range(i):
            if is_support:
                # Với Support Zone: Lệnh Limit Mua bị kích hoạt sớm nếu có nến trước đó đâm thủng Edge Trên (upper_bin)
                if future_candles[prev_idx].low <= upper_bin:
                    is_available = False
                    break
            else:
                # Với Resistance Zone: Lệnh Limit Bán bị kích hoạt sớm nếu có nến trước đó vượt qua Edge Dưới (lower_bin)
                if future_candles[prev_idx].high >= lower_bin:
                    is_available = False
                    break

        if is_available:
            valid_zones.append((i, lower_bin, upper_bin, upper_bin - lower_bin))

    return valid_zones

class DatasetBuilder:
    def __init__(self, cfg: AppConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.ast_visitor = ASTVisitor(digit_pad=self.cfg.base.digit_pad)
        
        # Read scale file text and create dict scale
        scale_file = "scripts/scale_factor.txt"
        self.scale = {}

        with open(scale_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Bỏ qua các dòng trống
                if not line or ":" not in line:
                    continue
                    
                key, value = line.split(":", 1)
                self.scale[key.strip()] = float(value.strip())
        
    def _find_zone(
        self,
        chart: ChartNode,
        future_candles: List[CandleNode],
    )-> List[ProgramNode]:
        results: List[ProgramNode] = []
        reward = TLangReward(self.cfg)
        noise = int((self.cfg.base.zone_width_max_bins - self.cfg.base.zone_width_min_bins)/3)
        for mode, zone_direction in [("support", "UP"), ("resistance", "DOWN")]:
            zones = find_truly_valid_zones(
                future_candles,
                mode,
                swing_window = 5,
                zone_width = self.cfg.base.zone_width_min_bins,      # Hard Config Range Min
                noise = noise,
                max_bin = self.cfg.base.bin_max,
            )
            for future_idx, sup_low, sup_high, width in zones:
                zone = ZoneNode(direction=zone_direction, lower_bin=sup_low, upper_bin=sup_high)
                
                score = reward.zone_score(zone, future_bins=future_candles)
                
                if score.zone_quality > 0.3:
                    if mode == "support":
                        trend = "UP"
                    else:
                        trend = "DOWN"
                    program: ProgramNode = ProgramNode(
                        chart=chart,
                        think=ThinkNode(
                            trend=trend,
                            current_price_bin=chart.current_price,
                            zone=zone
                        )
                    )
                    results.append(program)
                elif score.zone_quality > 0.2:
                    trend = "RANGE"
                    program: ProgramNode = ProgramNode(
                        chart=chart,
                        think=ThinkNode(
                            trend=trend,
                            current_price_bin=chart.current_price,
                            zone=zone
                        )
                    )
                    results.append(program)
        if len(results) == 0: # giá trong range, ko có zone nào giá trị
            program: ProgramNode = ProgramNode(
                chart=chart,
                think=ThinkNode(
                    trend="RANGE",
                    current_price_bin=chart.current_price,
                )
            )
            results.append(program)

        return results

    def _build_a_program(
        self,
        scale: float,
        input_window: np.ndarray,
        future_window: np.ndarray,
        atr_100: float,
    ) -> List[ProgramNode]:
        codec = ChartCodec(scale=scale, n_bins=self.cfg.base.n_bins)
        input_candles, open_anchor = codec._encode_input(input_window, atr_100)
        future_candles = codec._encode_future(future_window, open_anchor, atr_100)
        chart = ChartNode(candles=input_candles)
        programs = self._find_zone(chart, future_candles)

        return programs, future_candles
        
    def build_rows(
        self,
        symbol: str,
        input_window: np.ndarray,
        future_window: np.ndarray,
        atr_100: float,
        n_augments: int = 2,
    ):
        scale = self.scale[symbol]
        programs, future_candles = self._build_a_program(scale, input_window, future_window, atr_100)
        chart = programs[0].chart
        rows = []
        for program in programs:
            prompt = self.ast_visitor.render_chart_block(chart.candles)
            completion = self.ast_visitor.build_completion(program.think)
            row = {
                "symbol": symbol,
                "prompt": prompt,
                "completion": completion,
                "future_bins": [[int(c.open), int(c.high), int(c.low), int(c.close)] for c in future_candles],
            }
            rows.append(row)
        if n_augments > 0:
            chart_high = max(c.high for c in chart.candles)
            chart_low = min(c.low for c in chart.candles)
            chart_range = chart_high - chart_low
            if chart_range < self.cfg.base.n_bins * 0.5:
                aug_scales = [1.1, 1.2, 1.3, 1.4]
            elif chart_range > self.cfg.base.n_bins * 0.8:
                aug_scales = [0.9, 0.8, 0.7, 0.6]
            else:
                aug_scales = [0.8, 0.9, 1.1, 1.2]
                
            for _ in range(n_augments):
                aug_scale = self.rng.choice(aug_scales) / scale
                aug_programs, aug_future_candles = self._build_a_program(aug_scale, input_window, future_window, atr_100)
                for aug_program in aug_programs:
                    prompt = self.ast_visitor.render_chart_block(aug_program.chart.candles)
                    completion = self.ast_visitor.build_completion(aug_program.think)
                    row = {
                        "symbol": symbol,
                        "prompt": prompt,
                        "completion": completion,
                        "future_bins": [[int(c.open), int(c.high), int(c.low), int(c.close)] for c in aug_future_candles],
                    }
                    rows.append(row)
        
        return rows
                
    
import re

def extract_symbol(s):
    # Tìm chuỗi dạng: 6 chữ cái + _ + số + chữ (ví dụ EURUSD_1Min)
    match = re.search(r'[A-Z]{6}_\w+', s)
    return match.group(0) if match else s

def main(
    cfg: AppConfig, 
    input_dir: str,
    output_dir: str,
    seed: Optional[int] = None,
    n_augments = 2,
):
    from datasets import load_dataset
    import hashlib

    data_files = {
        "train": f"{input_dir}/slide_window_200_train.parquet",
        "val": f"{input_dir}/slide_window_200_val.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    dataset_builder = DatasetBuilder(cfg, seed=seed)
    
    def preprocess_for_llm(batch):
        prompts = []
        completions = []
        future_bins_list = []
        symbols = []
        
        batch_size = len(batch["symbol"])
        
        # Duyệt qua các phần tử trong batch (chạy trong RAM của batch đó, cực nhẹ)
        for i in range(batch_size):
            symbol = extract_symbol(batch["symbol"][i])
            input_window = np.array(batch["input_window"][i], dtype=np.float32)
            future_window = np.array(batch["future_window"][i], dtype=np.float32)
            atr_100 = batch["atr_100"][i]
            
            records = dataset_builder.build_rows(symbol, input_window, future_window, atr_100, n_augments=n_augments)
                        
            for record in records:
                prompts.append(record["prompt"])
                completions.append(record["completion"])
                future_bins_list.append(record["future_bins"])
                symbols.append(record["symbol"])
                
        # Trả về các cột mới cho Dataset LLM
        return {
            "prompt": prompts,
            "completion": completions,
            "future_bins": future_bins_list,
            "symbol": symbols,
        }
        
    llm_dataset = dataset.map(
        preprocess_for_llm,
        batched=True,
        batch_size=2000, # Mỗi lần nạp 2000 dòng vào RAM để parse
        num_proc=2,      # Số lượng nhân CPU chạy song song
        remove_columns=dataset["train"].column_names # Xóa các cột gốc (id, type, score...) để thu gọn dataset
    )
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    llm_dataset["train"].shuffle(seed=seed).to_parquet(f"{output_dir}/train_llm.parquet")
    llm_dataset["val"].to_parquet(f"{output_dir}/val_llm.parquet")    

if __name__ == '__main__':
    cfg: AppConfig = load_config("configs")
    main(cfg, "data/slide_window", "data/dataset", seed=42, n_augments=0)