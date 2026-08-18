import pandas as pd
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List

class SlideWindowGen:
    def __init__(
        self,
        file: str,
        input_window: int = 100,
        future_window: int = 100,
        batch_size: int = 2000,
        stride: int = 1,
        desc: str = "Processing" # Thêm tham số này để tqdm hiển thị tên file/symbol
    ):
        self.df = pd.read_csv(file)
        self.symbol = file.split("/")[-1].split(".")[0]
        self.batch_size = batch_size
        self.input_window = input_window
        self.future_window = future_window
        self.window = input_window + future_window
        self.stride = stride
        self.desc = desc

        self._length = (len(self.df) - self.window + 1) // stride
        self.data = self.df[["Open", "High", "Low", "Close"]].values
        self.atr_100 = self.df["ATR_100"].values

    def __iter__(self):
        batch = []

        for count in tqdm(range(self._length), desc=self.desc, leave=False, unit="win"):
            i = count * self.stride

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
            
class ParquetWriterUtil:
    def __init__(self, output_path: str, schema=None):
        self.output_path = output_path
        self.schema = schema
        self.writer = None

    # Sử dụng Context Manager (with) để đảm bảo file luôn được đóng an toàn
    def __enter__(self):
        self.writer = pq.ParquetWriter(self.output_path, self.schema, compression='snappy')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_batch(self, batch: List[Dict]):
        if not batch:
            return
        # Ghi data trực tiếp
        table = pa.Table.from_pylist(batch, schema=self.schema)
        self.writer.write_table(table)

    def close(self):
        if self.writer:
            self.writer.close()
            
def generate(inputs: List[tuple], **kwargs) -> None:
    for symbol, timeframe, file in tqdm(inputs, desc="All Files", position=0):
        gen = SlideWindowGen(file, **kwargs)
        for batch in gen:
            yield batch
            
def gen_validation_files():
    import glob

    folder = "data/preprocessed/val"
    all_files = glob.glob(folder + "/*.csv")

    schema = pa.schema([
        ("symbol", pa.string()),
        ("input_window", pa.list_(pa.list_(pa.float32()))),
        ("future_window", pa.list_(pa.list_(pa.float32()))),
        ("atr_100", pa.float32())
    ])

    total_rows = 0
    output_path = "data/slide_window/slide_window_200_val.parquet"
    # Bọc ParquetWriter vào context block để tự động flush và close khi hoàn tất
    with ParquetWriterUtil(output_path, schema=schema) as writer:
        for file in tqdm(all_files, desc="All Files", position=0):
            print(f"Đang xử lý: {file}")

            gen = SlideWindowGen(
                file=file,
                input_window=100,
                future_window=100,
                batch_size=2000,
                stride=1,
                desc=""
            )

            for batch in gen:
                writer.write_batch(batch)
                total_rows += len(batch)

            print(f"Đã ghi xong {file}. Tổng số dòng hiện tại: {total_rows}")

    print(f"Thành công! Toàn bộ quá trình hoàn tất. Có tổng cộng {total_rows} dòng.")
    
if __name__ == "__main__":
    gen_validation_files()