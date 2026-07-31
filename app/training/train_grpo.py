from __future__ import annotations

import argparse
import glob
import logging
import os

logger = logging.getLogger("train_grpo")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from app.config.schema import AppConfig

def _seed_from_round_id(round_id: str) -> int:
    import hashlib
    return int(hashlib.md5(round_id.encode()).hexdigest(), 16) % (2**31)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_size", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument(
        "--source_repo", default=None,
        help="Nguồn init NẾU round này chưa có checkpoint nào: SFT repo (round 1) "
             "hoặc checkpoint round liền trước (round N>1) — truyền tay, không tự động.",
    )
    p.add_argument("--output_dir", required=True)
    
    p.add_argument("--dataset_name", default=None, help="Dataset GRPO gốc (schema prompt/future_bins/symbol/window_id)")
    p.add_argument("--round_id", required=True, help="round id, vd: round1")

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
    
    if not args.repo_id or not args.dataset_name:
        print("--repo_id và --dataset_name bắt buộc khi train thật.")
        raise SystemExit(1)
    
    from app.config.loader import get_train_cfg
    from app.training.common import resolve_resume_checkpoint, print_device_info
    from app.tokenizer.hub import load_tokenizer
    from app.training.model.model_loader import ModelLoader
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
        
    os.makedirs(args.output_dir, exist_ok=True)
    print_device_info()
    
    push_to_hub = False
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        push_to_hub = True
    else:
        print("Có repo_id nhưng chưa có hf_token — nhớ gọi huggingface_hub.login() thủ công trước khi chạy.")

    # TODO: Init Round here
    round_config = cfg.rounds[args.round_id]
    print(round_config)
        
        # ------------------------------------------------------------
    # Tokenizer — giống hệt v1 (xem giải thích add_eos_token/add_bos_token
    # quirk trong train_grpo.py v1, không lặp lại).
    # ------------------------------------------------------------
    tok = load_tokenizer(repo_id=args.repo_id, allow_local_fallback=False)
    logger.info(f"tokenizer vocab_size = {tok.vocab_size}")
    tok.add_eos_token = False
    tok.add_bos_token = True
    
    # ------------------------------------------------------------
    # Model — 
    # ------------------------------------------------------------
    if args.source_repo is None:
        print("Chưa cố source_repo — yêu cầu set --source_repo.")
        exit(1)
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.repo_id)
    model_loader = ModelLoader(cfg.models, args.model_size)
    model = model_loader.build_continue_model(resume_checkpoint, args.source_repo)
    
    logger.info(f"model vocab_size = {model.config.vocab_size}")
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError(
            f"model.vocab_size ({model.config.vocab_size}) != tokenizer.vocab_size ({tok.vocab_size}) — "
            f"configs/models.yaml.vocab_size dang lech voi tokenizer that dang dung "
            f"(repo_id={args.repo_id!r}). Sua lai configs/models.yaml hoac kiem tra dung tokenizer_repo."
        )
    
    raw = load_dataset(args.dataset_name, split=args.train_split)
    
    train_cfg = get_train_cfg(cfg, "grpo")
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=_seed_from_round_id(args.round_id),
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        max_completion_length=args.max_completion_length,

        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        min_p=args.min_p if args.min_p > 0 else None,
        repetition_penalty=args.repetition_penalty,

        num_generations=args.num_generations,
        use_vllm=args.use_vllm,
        fp16=args.fp16,
        bf16=not args.fp16,
        push_to_hub=push_to_hub,
        hub_model_id=args.repo_id if push_to_hub else None,
        hub_strategy="checkpoint" if push_to_hub else "every_save",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
    )
    
    
    pass

if __name__ == "__main__":
    from app.config.loader import load_config
    
    cfg : AppConfig = load_config("configs")
    main(cfg)