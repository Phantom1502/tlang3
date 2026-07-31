file = "data/pretrain/zone_grpo_val.parquet"

import pandas as pd

df = pd.read_parquet(file)

print(df.iloc[1]['prompt'])
print(df.iloc[1]['future_bins'])
print(df.iloc[1]['symbol'])
print(df.iloc[1]['window_id'])
