"""
Run examples:

python -m scripts.eval --model_repo sullivan1502/base-zone-grpo --round_id round1 --dataset_repo sullivan1502/zone-grpo-data
"""
from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("scripts.eval")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from app.config.schema import AppConfig, RoundConfig
from app.training.zone_eval import ZoneEval

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_repo", required=True, help="checkpoint task1 (HF Hub repo_id)")
    p.add_argument("--round_id", required=True, help="round id, vd: round1")
    
    p.add_argument("--dataset_repo", required=True, help="Dataset GRPO gốc (schema prompt/future_bins/symbol/window_id)")
    p.add_argument("--subfolder", default=None, help="Subfolder trong dataset repo. Ex: 'last-checkpoint' (hub_strategy='checkpoint' của train_grpo.py push vào đây, KHÁC bản 'final' ở root repo). Default: None (root repo).")
    p.add_argument("--revision", default=None, help="git revision/tag/commit trên Hub — dùng khi cần pin đúng 1 lần push cụ thể (KHÁC subfolder — 1 repo có thể có nhiều revision VÀ nhiều subfolder cùng lúc, tuỳ cách bạn tổ chức checkpoint). Default: None (latest).")
    p.add_argument("--split", default="val", help="Split trong dataset repo. Default: 'val'.")

    p.add_argument("--tokenizer_repo", default=None, help="Tokenizer repo_id (HF Hub). Default: None (dùng tokenizer trong model_repo).")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=24)
    
    p.add_argument("--do_sample", type=bool, default=False)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    
    p.add_argument("--limit", type=int, default=None)
    
    return p

def main(cfg: AppConfig):
    args = build_arg_parser().parse_args()
    
    zone_eval = ZoneEval(
        cfg=cfg,
        round_id=args.round_id,
        model_repo=args.model_repo,
        dataset_repo=args.dataset_repo,
        revision=args.revision,
        subfolder=args.subfolder,
        split=args.split,
        tokenizer_repo=args.tokenizer_repo,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
    )
    zone_eval.run()

if __name__ == "__main__":   
    from app.config.loader import load_config
        
    cfg : AppConfig = load_config("configs")
    main(cfg)