from typing import Optional,List
from app.config import AppConfig, load_config
from app.lang import Parser, ParseResult, ProgramNode, ChartNode, CandleNode, ThinkNode, ZoneNode
from collections import defaultdict, Counter
from app.candle import Candle, augment_shift
from app.lang.ast_visitor import ASTVisitor
import random
def main(
    cfg: AppConfig, 
    data_repo: str,
    output_dir: str,
    seed: Optional[int] = None
):
    from datasets import load_dataset
    dataset = load_dataset(data_repo)
    ast_visitor = ASTVisitor(cfg.base.digit_pad)
    
    def preprocess_for_llm(batch):
        prompts = []
        future_bins_list = []
        symbols = []
        window_ids = []
        
        batch_size = len(batch["prompt"])
        rng = random.Random(seed)
        
        # Duyệt qua các phần tử trong batch (chạy trong RAM của batch đó, cực nhẹ)
        for i in range(batch_size):
            prompt = batch["prompt"][i]
            future_bins = batch["future_bins"][i]
            symbol = batch["symbol"][i]
            window_id = batch["window_id"][i]
            
            parse_result: ParseResult = Parser.from_text(cfg, prompt).parse()
            first_candle = parse_result.ast.chart.candles[0]
            
            if first_candle.open == 1024:
                future_candles: List[Candle] = [Candle(open=b[0], high=b[1], low=b[2], close=b[3]) for b in future_bins]
                full_chart: List[Candle] = parse_result.ast.chart.candles + future_candles
                shifted = augment_shift(full_chart, rng=rng, n_bins=cfg.base.n_bins)   # augment CẢ window (input+future)
                if shifted is not None:
                    input_candles = shifted[:cfg.window.input_candles]
                    future_candles = shifted[cfg.window.input_candles:]
                    prompt_aug = ast_visitor.render_chart_block(input_candles)
                    future_bins_aug = [[c.open, c.high, c.low, c.close] for c in future_candles]
                    window_id_aug = f"{window_id}_aug{seed}"
                    
                    #print(f"Prompt: {prompt}\nFuture: {future_bins}\nAugmented prompt: {prompt_aug}\nAugmented future: {future_bins_aug}")
                    
                    prompts.append(prompt_aug)
                    future_bins_list.append(future_bins_aug)
                    symbols.append(symbol)
                    window_ids.append(window_id_aug)
            else:
                prompts.append(prompt)
                future_bins_list.append(future_bins)
                symbols.append(symbol)
                window_ids.append(window_id)
        
        return {"prompt": prompts, "future_bins": future_bins_list, "symbol": symbols, "window_id": window_ids}
                    
    llm_dataset = dataset.map(
        preprocess_for_llm,
        batched=True,
        batch_size=2000, # Mỗi lần nạp 2000 dòng vào RAM để parse
        num_proc=4,      # Số lượng nhân CPU chạy song song
        remove_columns=dataset["train"].column_names # Xóa các cột gốc (id, type, score...) để thu gọn dataset
    )
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    llm_dataset["train"].shuffle(seed=seed).to_parquet(f"{output_dir}/zone_grpo_train.parquet")
    llm_dataset["val"].to_parquet(f"{output_dir}/zone_grpo_val.parquet")  
    
if __name__ == "__main__":
    cfg : AppConfig = load_config("configs")
    main(
        cfg, 
        "sullivan1502/zone-grpo-data",
        "data/re_aug_grpo",
    )  