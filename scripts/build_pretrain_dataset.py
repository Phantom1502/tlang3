import pandas as pd
import numpy as np

from typing import List
import pyarrow as pa
from app.data_prepare.candle import Candle
from app.config.loader import load_config, get_scale
from app.config.schema import AppConfig, WindowConfig
from app.data_prepare.chartcodec import ChartCodec
from app.data_prepare.dataset_builder import DatasetBuilder
from app.utils.parquet_writer import ParquetWriterUtil
from tqdm.auto import tqdm  # Thêm dòng này

class ZonePretrainOneFileGen:
    def __init__(
        self, 
        cfg, 
        codec, 
        builder, 
        file: str, 
        samples_per_chart: int = 4,
        n_augments: int = 0,
        batch_size: int = 2000, 
        stride: int = 1,
        desc: str = "Processing" # Thêm tham số này để tqdm hiển thị tên file/symbol
    ):
        self.cfg = cfg
        self.df = pd.read_csv(file)
        self.codec = codec
        self.builder = builder
        self.batch_size = batch_size
        self.samples_per_chart = samples_per_chart
        self.n_augments = n_augments
        self.stride = stride
        self.desc = desc
        
        self._length = (len(self.df) - self.cfg.window.window_size + 1) // stride

    def __iter__(self):
        # TỐI ƯU HÓA: Chuyển các cột hay truy cập sang Numpy Array để truy xuất O(1)
        # Nhanh hơn 50-100x so với dùng self.df.loc[...] trong vòng lặp
        opens = self.df["Open"].values
        atrs = self.df["ATR_100"].values
        
        batch = []
        
        for count in tqdm(range(self._length), desc=self.desc, leave=False, unit="win"):
            i = count * self.stride
            anchor_open = opens[i]
            anchor_atr = atrs[i]
            
            if anchor_atr <= 0 or np.isnan(anchor_atr):
                continue
                
            # Cắt cửa sổ (pandas slice still takes a little time, but acceptable)
            window = self.df.iloc[i : i + self.cfg.window.window_size]
            candles = self.codec.encode_window(window, anchor_open, anchor_atr)
            
            rows = self.builder.build_pretrain_rows(candles, self.samples_per_chart, self.n_augments)
            batch.extend(rows)
            
            # Nếu mẻ dữ liệu đủ lớn, đẩy (yield) ra ngoài và reset lại mẻ
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                
        # TRÁNH MẤT DATA: Trả về phần dữ liệu còn sót lại chưa đủ batch_size
        if batch:
            yield batch
            
class ZonePretrainGen:
    def __init__(self, output_path: str) -> None:
        self.cfg = load_config("configs")
        self.builder = DatasetBuilder(self.cfg, seed=42)
        self.output_path = output_path
        
        self.schema = pa.schema([
            ("prompt", pa.string()),
            ("completion", pa.string()),
        ])
                
    def generate(
        self, 
        inputs: List[tuple],
        samples_per_chart: int = 1,
        n_augments: int = 5,
        stride: int = 1
    ):
        total_rows = 0
        
        # Bọc ParquetWriter vào context block để tự động flush và close khi hoàn tất
        with ParquetWriterUtil(self.output_path, schema=self.schema) as writer:
            for symbol, timeframe, file in tqdm(inputs, desc="All Files", position=0):
                print(f"Đang xử lý: {symbol} {timeframe}")
                scale = get_scale(self.cfg, symbol, timeframe)
                codec = ChartCodec(scale=scale, n_bins=self.cfg.base.n_bins)
                
                gen = ZonePretrainOneFileGen(
                    self.cfg, codec, self.builder, file, 
                    samples_per_chart=samples_per_chart, 
                    n_augments=n_augments, 
                    stride=stride
                )
                
                for batch in gen:
                    writer.write_batch(batch)
                    total_rows += len(batch)
                
                print(f"Đã ghi xong {symbol}. Tổng số dòng hiện tại: {total_rows}")
                
        print(f"Thành công! Toàn bộ quá trình hoàn tất. Có tổng cộng {total_rows} dòng.")
        
if __name__ == "__main__":
    output_path = "data/pretrain/zone_pretrain_train.parquet"
    inputs = [
        ("XAUUSD", "M1", "data/preprocessed/train/XAUUSD_1Min.csv"),
        ("XAUUSD", "M5", "data/preprocessed/train/XAUUSD_5Min.csv"),
        ("XAUUSD", "M15", "data/preprocessed/train/XAUUSD_15Min.csv"),
        ("XAUUSD", "H1", "data/preprocessed/train/XAUUSD_H1.csv"),
        ("XAUUSD", "Daily", "data/preprocessed/train/XAUUSD_Daily.csv"),
        ("EURUSD", "M1", "data/preprocessed/train/EURUSD_1Min.csv"),
        ("EURUSD", "M5", "data/preprocessed/train/EURUSD_5Min.csv"),
        ("EURUSD", "M15", "data/preprocessed/train/EURUSD_15Min.csv"),
        ("EURUSD", "H1", "data/preprocessed/train/EURUSD_H1.csv"),
        ("GBPUSD", "M1", "data/preprocessed/train/GBPUSD_1Min.csv"),
        ("GBPUSD", "M5", "data/preprocessed/train/GBPUSD_5Min.csv"),
        ("GBPUSD", "M15", "data/preprocessed/train/GBPUSD_15Min.csv"),
        ("GBPUSD", "H1", "data/preprocessed/train/GBPUSD_H1.csv"),
        ("US500", "M1", "data/preprocessed/train/US500_1Min.csv"),
        ("US500", "M5", "data/preprocessed/train/US500_5Min.csv"),
        ("US500", "M15", "data/preprocessed/train/US500_15Min.csv"),
        ("US500", "H1", "data/preprocessed/train/US500_H1.csv"),
    ]
    gen = ZonePretrainGen(output_path)
    gen.generate(
        inputs,
        samples_per_chart=1,
        n_augments=4,
        stride=10
    )