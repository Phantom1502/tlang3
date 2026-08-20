from __future__ import annotations

import numpy as np
import random
from typing import List, Optional, Tuple, Literal

from app.config import AppConfig, load_config, get_scale
from app.training.reward import TLangReward
from tlang import (
    ChartCodec,
    ASTVisitor,
    ProgramNode,
    ChartNode,
    ThinkNode,
    CandleNode,
    ZoneNode
)

from collections import Counter

def find_truly_valid_zones(
    last_n_candles: List[CandleNode],
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
    last_n = len(last_n_candles)
    candles = last_n_candles + future_candles
    n = len(candles)
    
    if n == 0:
        return valid_zones

    is_support = (mode == "support")

    for i in range(n):
        # 1. KIỂM TRA ĐIỀU KIỆN SWING HIGH / SWING LOW
        # Đảm bảo đủ số nến swing_window ở hai bên
        left_start = max(0, i - swing_window)
        right_end = min(n - 1, i + swing_window)

        if is_support:
            current_val = candles[i].low
            # Là Swing Low nếu giá Low hiện tại <= tất cả các nến trong cửa sổ xung quanh
            is_swing = all(current_val <= candles[j].low for j in range(left_start, right_end + 1) if j != i)
        else:
            current_val = candles[i].high
            # Là Swing High nếu giá High hiện tại >= tất cả các nến trong cửa sổ xung quanh
            is_swing = all(current_val >= candles[j].high for j in range(left_start, right_end + 1) if j != i)

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
        if i - last_n > 0:
            for prev_idx in range(i - last_n):
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
    def __init__(
        self, 
        cfg: AppConfig,
        seed: Optional[int] = None
    ):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.ast_visitor = ASTVisitor(digit_pad=self.cfg.base.digit_pad)
        self.counter = Counter()
        
    def _find_zone(
        self,
        chart: ChartNode,
        future_candles: List[CandleNode],
        trend_threshhold: float = 0.4,
        hold_threshhold: float = 0.2,
        swing_window: int = 5,
        is_noise: bool = True
    )-> List[ProgramNode]:
        last_n_candles = chart.candles[-self.cfg.base.last_n_candles:]
        results: List[ProgramNode] = []
        reward = TLangReward(self.cfg)
        noise = 0
        if is_noise:
            noise = int((self.cfg.base.zone_width_max_bins - self.cfg.base.zone_width_min_bins)/3)
        for zone_direction in ["support", "resistance"]:
            zones = find_truly_valid_zones(
                last_n_candles,
                future_candles,
                zone_direction,
                swing_window = swing_window,
                zone_width = self.cfg.base.zone_width_min_bins,      # Hard Config Range Min
                noise = noise,
                max_bin = self.cfg.base.bin_max,
            )
            for future_idx, lower_bin, upper_bin, width in zones:
                zone = ZoneNode(
                    direction=zone_direction, 
                    lower_bin=lower_bin, 
                    upper_bin=upper_bin
                )
                
                score = reward.zone_score(zone, future_bins=future_candles)
                
                if score.zone_quality > trend_threshhold:
                    if zone_direction == "support":
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
                    self.counter[f"{program.think.trend}_{program.think.zone.direction}"] += 1
                    results.append(program)
                elif score.zone_quality > hold_threshhold:
                    program: ProgramNode = ProgramNode(
                        chart=chart,
                        think=ThinkNode(
                            trend="RANGE",
                            current_price_bin=chart.current_price,
                            zone=zone
                        )
                    )
                    self.counter[f"{program.think.trend}_{program.think.zone.direction}"] += 1
                    results.append(program)
                    
        if len(results) == 0:
            program: ProgramNode = ProgramNode(
                chart=chart,
                think=ThinkNode(
                    trend="RANGE",
                    current_price_bin=chart.current_price,
                )
            )
            self.counter[f"{program.think.trend}_NOZONE"] += 1
            results.append(program)
        return results
    
    def _build_a_program(
        self,
        scale: float,
        input_window: np.ndarray,
        future_window: np.ndarray,
        atr_100: float,
        trend_threshhold: float = 0.6,
        hold_threshhold: float = 0.3,
        swing_window: int = 5,
        is_noise: bool = True
    ) -> List[ProgramNode]:
        codec = ChartCodec(scale=scale, n_bins=self.cfg.base.n_bins)
        input_candles, open_anchor = codec._encode_input(input_window, atr_100)
        future_candles = codec._encode_future(future_window, open_anchor, atr_100)
        chart = ChartNode(candles=input_candles)
        programs = self._find_zone(
            chart=chart, 
            future_candles=future_candles,
            trend_threshhold=trend_threshhold,
            hold_threshhold=hold_threshhold,
            swing_window=swing_window,
            is_noise=is_noise
        )

        return programs, future_candles
    
    def build_pretrain_rows(
        self,
        symbol_timeframe: str,
        input_window: np.ndarray,
        future_window: np.ndarray,
        atr_100: float,
        trend_threshhold: float = 0.6,
        hold_threshhold: float = 0.3,
        swing_window: int = 5,
    ):
        symbol = symbol_timeframe.split("_")[0]
        timeframe = symbol_timeframe.split("_")[1]
        scale = get_scale(self.cfg, symbol, timeframe)
        
        programs, future_candles = self._build_a_program(
            scale=scale,
            input_window=input_window,
            future_window=future_window,
            atr_100=atr_100,
            trend_threshhold=trend_threshhold,
            hold_threshhold=hold_threshhold,
            swing_window=swing_window,
            is_noise=True
        )
        
        chart = programs[0].chart
        rows = []
        for program in programs:
            prompt = self.ast_visitor.render_chart_block(chart.candles)
            completion = self.ast_visitor.build_completion(program.think)
            row = {
                "symbol": symbol_timeframe,
                "prompt": prompt,
                "completion": completion,
            }
            rows.append(row)

        return rows
    
def build_pretrain_dataset(
    cfg: AppConfig,
    input_dir: str,
    output_dir: str,
    seed: Optional[int] = None,
    trend_threshhold: float = 0.6,
    hold_threshhold: float = 0.3,
    swing_window: int = 5,
):
    from datasets import load_dataset
    import os
    
    data_files = {
        "train": f"{input_dir}/window_200_train_*.parquet",
        "val": f"{input_dir}/window_200_val.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    dataset_builder = DatasetBuilder(cfg, seed=seed)
    
    def preprocess_for_llm(batch):
        prompts = []
        completions = []
        symbols = []
        
        batch_size = len(batch["symbol"])
        
        # Duyệt qua các phần tử trong batch (chạy trong RAM của batch đó, cực nhẹ)
        for i in range(batch_size):
            symbol = batch["symbol"][i]
            input_window = np.array(batch["input_window"][i], dtype=np.float32)
            future_window = np.array(batch["future_window"][i], dtype=np.float32)
            atr_100 = batch["atr_100"][i]
            
            records = dataset_builder.build_pretrain_rows(
                symbol, 
                input_window, 
                future_window, 
                atr_100,
                trend_threshhold=trend_threshhold,
                hold_threshhold=hold_threshhold,
                swing_window=swing_window
            )
                        
            for record in records:
                prompts.append(record["prompt"])
                completions.append(record["completion"])
                symbols.append(record["symbol"])
                
        print(dataset_builder.counter)
                
        # Trả về các cột mới cho Dataset LLM
        return {
            "prompt": prompts,
            "completion": completions,
            "symbol": symbols,
        }
        
    llm_dataset = dataset.map(
        preprocess_for_llm,
        batched=True,
        batch_size=2000, # Mỗi lần nạp 2000 dòng vào RAM để parse
        num_proc=os.cpu_count(),      # Số lượng nhân CPU chạy song song
        remove_columns=dataset["train"].column_names # Xóa các cột gốc (id, type, score...) để thu gọn dataset
    )
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    llm_dataset["train"].shuffle(seed=seed).to_parquet(f"{output_dir}/train_pretrain.parquet")
    llm_dataset["val"].to_parquet(f"{output_dir}/val_pretrain.parquet") 
    
if __name__ == "__main__":
    cfg: AppConfig = load_config("configs")
    build_pretrain_dataset(cfg, "data/dataset", "data/dataset")