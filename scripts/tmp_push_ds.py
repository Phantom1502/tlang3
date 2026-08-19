import os
import glob
from datasets import Dataset, DatasetDict, concatenate_datasets

def _load_and_shuffle_split(folder_path: str, split_name: str, file_name: str = "parquet-train-*.arrow", seed: int = 42) -> Dataset:
    """Hàm phụ trợ: Load tất cả file Arrow trong 1 folder, concat và shuffle."""
    arrow_files = sorted(glob.glob(os.path.join(folder_path, file_name), recursive=True))
    if not arrow_files:
        raise FileNotFoundError(f"❌ Không tìm thấy tệp .arrow nào trong folder {split_name}: {folder_path}")

    print(f"📦 [{split_name.upper()}] Tìm thấy {len(arrow_files)} tệp Arrow. Đang đọc...")
    
    # Memory-map từng file (Zero-copy, không tốn RAM)
    datasets_list = [Dataset.from_file(f) for f in arrow_files]
    
    # Ghép các shards lại
    ds = concatenate_datasets(datasets_list)
    print(f"✅ [{split_name.upper()}] Load thành công! Tổng số dòng: {len(ds):,}")

    # Shuffle
    print(f"🔀 [{split_name.upper()}] Đang shuffle (seed={seed})...")
    ds = ds.shuffle(seed=seed)
    
    return ds


def push_arrow_splits_to_hub(
    train_folder: str,
    val_folder: str = None,
    repo_id: str = None,
    token: str = None,
    private: bool = True,
    seed: int = 42,
    max_shard_size: str = "300MB"  # 300MB giúp streaming & resume nhanh hơn rất nhiều so với 1GB
):
    """
    Load dữ liệu Train và Validation từ 2 thư mục Arrow riêng biệt,
    đóng gói thành DatasetDict và push trực tiếp lên Hugging Face Hub.
    """
    dataset_dict = {}

    # 1. Xử lý Train Split
    dataset_dict["train"] = _load_and_shuffle_split(train_folder, "train", file_name="cache-addafc6494b6b8fa_*.arrow", seed=seed)

    # 2. Xử lý Validation Split (nếu có)
    if val_folder and os.path.exists(val_folder):
        dataset_dict["val"] = _load_and_shuffle_split(val_folder, "val", file_name="cache-8dbef44d421e3629_*.arrow", seed=seed)
    else:
        print("⚠️ Không tìm thấy val_folder hoặc không được truyền vào. Chỉ push split 'train'.")

    # 3. Đóng gói thành DatasetDict
    final_ds = DatasetDict(dataset_dict)
    print(f"\n📊 Cấu trúc Dataset hoàn chỉnh:\n{final_ds}")

    # 4. Push trực tiếp lên Hugging Face Hub
    print(f"\n🚀 Đang stream & push dataset lên HF Hub: '{repo_id}'...")
    final_ds.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
        max_shard_size=max_shard_size
    )

    print(f"\n🎉 HOÀN THÀNH! Dataset của bạn đã sẵn sàng tại: https://huggingface.co/datasets/{repo_id}")

push_arrow_splits_to_hub(
    train_folder=r"C:\Users\sulli\.cache\huggingface\datasets\parquet\default-e5cf23ce92d05505\0.0.0\a859584b6747f312c02efa800dcc0a33038f602cb670b65dc6103d3f269c5565",        # Đường dẫn tới folder arrow của Train
    val_folder=r"C:\Users\sulli\.cache\huggingface\datasets\parquet\default-e5cf23ce92d05505\0.0.0\a859584b6747f312c02efa800dcc0a33038f602cb670b65dc6103d3f269c5565",            # Đường dẫn tới folder arrow của Val
    repo_id="sullivan1502/zone-pretrain-dataset", # <--- Đổi tên repo HF của bạn
    private=True,
    seed=42,
    max_shard_size="1000MB"                     # Khuyên dùng 300MB - 500MB cho streaming
)