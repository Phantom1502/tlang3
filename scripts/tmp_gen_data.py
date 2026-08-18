from tlang import (
    ChartCodec,
    CandleNode,
    ChartNode,
    ThinkNode,
    ZoneNode,
    ProgramNode,
    TLangConfig,
    plot_program
)
from .tmp_dataset_build import DatasetBuilder
from app.config import AppConfig, load_config
from datasets import load_dataset
import numpy as np

import re

def extract_symbol(s):
    # Tìm chuỗi dạng: 6 chữ cái + _ + số + chữ (ví dụ EURUSD_1Min)
    match = re.search(r'[A-Z]{6}_\w+', s)
    return match.group(0) if match else s

cfg: AppConfig = load_config("./configs")

data = load_dataset("parquet", data_files="data/slide_window/slide_window_200_train.parquet", split="train")

dataset = DatasetBuilder(cfg,seed=42)
records = dataset.build_rows(
    extract_symbol(data["symbol"][0]),
    np.array(data["input_window"][0], dtype=np.float32),
    np.array(data["future_window"][0], dtype=np.float32),
    data["atr_100"][0],
    n_augments=0
)
for record in records:
    print(record["prompt"])
    print(record["completion"])
    print(record["future_bins"])

