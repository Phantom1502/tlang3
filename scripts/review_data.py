file = "data/pretrain/zone_pretrain_train.parquet"

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

table = pq.read_table(file)
df = table.to_pandas()

print(df.iloc[0]["prompt"])
print(df.iloc[0]["completion"])