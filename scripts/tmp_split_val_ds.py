from datasets import load_dataset

dataset = load_dataset("parquet", data_files="data/dataset/val_zone.parquet", split="train")
print(dataset.shape)

# select 1000 item with random
split_ds = dataset.shuffle(seed=42).select(range(1000))
print(split_ds.shape)

split_ds.to_parquet("data/dataset/val_zone_1000.parquet")