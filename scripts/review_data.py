file = "data/pretrain/zone_grpo_val.parquet"

import pandas as pd

df = pd.read_parquet(file)

print(df.iloc[0]['prompt'])