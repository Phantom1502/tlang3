import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Iterator, List
from pathlib import Path
import math

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
      
def split_parquet(input_file: str, num_splits: int = 10):  
    file_name = Path(input_file).stem
    folder = Path(input_file).parent
    
    parquet_file = pq.ParquetFile(input_file)
    total_row_groups = parquet_file.num_row_groups
    total_rows = parquet_file.metadata.num_rows

    print(
        f"Tổng số dòng: {total_rows:,} | Tổng số Row Groups: {total_row_groups}"
    )

    # Tính số row groups cho mỗi file
    groups_per_split = math.ceil(total_row_groups / num_splits)

    for i in range(num_splits):
        start_rg = i * groups_per_split
        end_rg = min((i + 1) * groups_per_split, total_row_groups)

        if start_rg >= total_row_groups:
            break

        # Chỉ đọc các Row Group cần thiết vào bộ nhớ
        row_group_indices = list(range(start_rg, end_rg))
        table = parquet_file.read_row_groups(row_group_indices)

        output_file = f"{folder}/{file_name}_part_{i+1:02d}.parquet"
        pq.write_table(table, output_file, compression="snappy")

        print(
            f"Đã tạo {output_file}: {table.num_rows:,} dòng (Row groups {start_rg} -> {end_rg-1})"
        )    