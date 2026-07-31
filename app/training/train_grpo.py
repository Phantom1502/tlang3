from __future__ import annotations

import argparse
import glob
import logging
import os

logger = logging.getLogger("train_grpo")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from app.config.schema import AppConfig

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument("--repo_id", default=None, help="Checkpoint repo của CHÍNH round này (resume-from/push-to)")
    p.add_argument(
        "--source_repo", default=None,
        help="Nguồn init NẾU round này chưa có checkpoint nào: SFT repo (round 1) "
             "hoặc checkpoint round liền trước (round N>1) — truyền tay, không tự động.",
    )
    p.add_argument("--output_dir", required=True)

    p.add_argument("--hf_token", default=None, help="HF Token")
    p.add_argument("--repo_id", required=True, help="Model Repo ID on HF Hub")

    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--bf16", dest="fp16", action="store_false")
    
    return p

def main(cfg: AppConfig):
    args = build_arg_parser().parse_args()
    
    if args.model_size not in cfg.models.presets:
        print(
            f"--model_size={args.model_size!r} khong co trong configs/models.yaml. "
            f"Cac preset hop le: {sorted(cfg.models.presets.keys())}"
        )
        raise SystemExit(1)
    
    from app.training.common import resolve_resume_checkpoint, print_device_info

    os.makedirs(args.output_dir, exist_ok=True)
    print_device_info()
    
    push_to_hub = False
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        push_to_hub = True
    else:
        print("Có repo_id nhưng chưa có hf_token — nhớ gọi huggingface_hub.login() thủ công trước khi chạy.")

    pass

if __name__ == "__main__":
    from app.config.loader import load_config
    
    cfg : AppConfig = load_config("configs")
    main(cfg)