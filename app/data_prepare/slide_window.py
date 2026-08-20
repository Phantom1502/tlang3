import pandas as pd
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List
from pathlib import Path
from ..utils.parquet_writer import ParquetWriterUtil, split_parquet

SCHEMA =pa.schema([
    ("symbol", pa.string()),
    ("input_window", pa.list_(pa.list_(pa.float32()))),
    ("future_window", pa.list_(pa.list_(pa.float32()))),
    ("atr_100", pa.float32())
])
class SlideWindowGen:
    def __init__(
        self,
        file: str,
        input_window: int = 100,
        future_window: int = 100,
        batch_size: int = 2000,
        stride: int = 1,
        shift: int = 0,
        desc: str = "Processing" # Thêm tham số này để tqdm hiển thị tên file/symbol
    ):
        self.df = pd.read_csv(file)
        self.symbol = Path(file).stem
        self.batch_size = batch_size
        self.input_window = input_window
        self.future_window = future_window
        self.window = input_window + future_window
        self.stride = stride
        self.shift = shift
        self.desc = desc

        self._length = (len(self.df) - self.window + 1) // stride
        self.data = self.df[["Open", "High", "Low", "Close"]].values
        self.atr_100 = self.df["ATR_100"].values

    def __iter__(self):
        batch = []

        for count in tqdm(range(self._length), desc=self.desc, leave=False, unit="win"):
            i = count * self.stride + self.shift

            # Cắt cửa sổ (pandas slice still takes a little time, but acceptable)
            input_window = self.data[i : i + self.input_window]
            future_window = self.data[i + self.input_window : i + self.window]
            atr_100 = self.atr_100[i]

            input_candles = []
            for row in input_window:
                o, h, l, c = row
                input_candles.append([float(o), float(h), float(l), float(c)])

            future_candles = []
            for row in future_window:
                o, h, l, c = row
                future_candles.append([float(o), float(h), float(l), float(c)])
            record = {
                "symbol": self.symbol,
                "input_window": input_candles,   # List 2D: [[...], [...]]
                "future_window": future_candles, # List 2D: [[...], [...]]
                "atr_100": float(atr_100)        # Giá trị float đơn lẻ
            }
            batch.append(record)

            # Nếu mẻ dữ liệu đủ lớn, đẩy (yield) ra ngoài và reset lại mẻ
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        # TRÁNH MẤT DATA: Trả về phần dữ liệu còn sót lại chưa đủ batch_size
        if batch:
            yield batch
            
def generate_slidewindow(
    folder: str,
    output_path: str,
    input_window: int = 100,
    future_window: int = 100,
    batch_size: int = 2000,
    stride: int = 1,
    shift: int = 0
):
    import glob

    all_files = glob.glob(folder + "/*.csv")
    total_rows = 0
    with ParquetWriterUtil(output_path, schema=SCHEMA) as writer:
        for file in tqdm(all_files, desc="All Files", position=0):
            gen = SlideWindowGen(
                file=file,
                input_window=input_window,
                future_window=future_window,
                batch_size=batch_size,
                stride=stride,
                shift=shift,
                desc=Path(file).stem
            )

            for batch in gen:
                writer.write_batch(batch)
                total_rows += len(batch)
                
            print(f"Đã ghi xong {file}. Tổng số dòng hiện tại: {total_rows}")
    print(f"Thành công! Toàn bộ quá trình hoàn tất. Có tổng cộng {total_rows} dòng.")
    
if __name__ == "__main__":
    from app.config import AppConfig, load_config
    cfg: AppConfig = load_config("configs")
    
    # Generate slide window for train pretrain
    generate_slidewindow(
        folder="data/preprocessed/train",
        output_path="data/slide_window/pretrain/window_200_train.parquet",
        input_window=cfg.window.input_candles,
        future_window=cfg.window.outcome_horizon,
        batch_size=2000,
        stride=3,
    )
    split_parquet(
        input_file="data/slide_window/pretrain/window_200_train.parquet",
        num_splits=10
    )
    
    # Generate slide window for val pretrain
    generate_slidewindow(
        folder="data/preprocessed/val",
        output_path="data/slide_window/pretrain/window_200_val.parquet",
        input_window=cfg.window.input_candles,
        future_window=cfg.window.outcome_horizon,
        batch_size=2000,
        stride=261,
    )
    
    # Generate slide window for train grpo
    generate_slidewindow(
        folder="data/preprocessed/train",
        output_path="data/slide_window/grpo/window_200_train.parquet",
        input_window=cfg.window.input_candles,
        future_window=cfg.window.outcome_horizon,
        batch_size=2000,
        stride=30,
        shift=1
    )
    split_parquet(
        input_file="data/slide_window/grpo/window_200_train.parquet",
        num_splits=10
    )
    
    # Generate slide window for val grpo
    generate_slidewindow(
        folder="data/preprocessed/val",
        output_path="data/slide_window/grpo/window_200_val.parquet",
        input_window=cfg.window.input_candles,
        future_window=cfg.window.outcome_horizon,
        batch_size=2000,
        stride=10,
    )