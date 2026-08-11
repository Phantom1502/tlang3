from typing import Optional
from app.config import AppConfig, load_config
from app.lang import Parser, ParseResult
from collections import defaultdict, Counter
def main(
    cfg: AppConfig, 
    input_dir: str,
):
    from datasets import load_dataset
    import hashlib

    data_files = {
        "train": f"{input_dir}/zone_sft_re_train_raw.parquet",
        "val": f"{input_dir}/zone_sft_re_val_raw.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)
    
    
    def preprocess_for_llm(batch):
        batch_size = len(batch["prompt"])
        counter = Counter()
        # Duyệt qua các phần tử trong batch (chạy trong RAM của batch đó, cực nhẹ)
        for i in range(batch_size):
            prompt = batch["prompt"][i]
            
            parse_result: ParseResult = Parser.from_text(cfg, prompt).parse()
            first_candle = parse_result.ast.chart.candles[0]
            counter[first_candle.open] += 1
        
        print(counter)
    llm_dataset = dataset.map(
        preprocess_for_llm,
        batched=True,
        batch_size=2000, # Mỗi lần nạp 2000 dòng vào RAM để parse
        num_proc=4,      # Số lượng nhân CPU chạy song song
        remove_columns=dataset["train"].column_names # Xóa các cột gốc (id, type, score...) để thu gọn dataset
    )
    
if __name__ == "__main__":
    cfg : AppConfig = load_config("configs")
    main(cfg, "data/pretrain")  