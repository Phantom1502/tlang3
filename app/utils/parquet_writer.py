import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Iterator, List

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