from datasets import load_dataset
from pathlib import Path

def limit_ds(ds_file: str, limit: int = 1000):
    file_name = Path(ds_file).stem
    file_path = Path(ds_file).parent
    dataset = load_dataset("parquet", data_files=ds_file, split="train")

    limit_ds = dataset.shuffle(seed=42).select(range(limit))

    limit_ds.to_parquet(f"{file_path}/{file_name}_{limit}.parquet")
    
    
if __name__ == "__main__":
    limit_ds("data/dataset/pretrain/val_zone_pretrain.parquet", limit=1000)