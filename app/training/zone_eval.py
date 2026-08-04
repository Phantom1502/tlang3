"""
app/training/eval/zone_eval.py — Eval độc lập cho task1 (zone-inference),
chạy trên val split, TÁI DÙNG nguyên TLangReward (cùng công thức reward
với train), thống kê 2 thành phần quan trọng:

    1. Tỉ lệ zone_type (NO_ZONE/SUP_ZONE/RES_ZONE) trong số completion đã
       pass gate — xem model có thiên vị 1 hướng bất thường không.
    2. Mean reward theo SUP_ZONE/RES_ZONE (NO_ZONE bỏ qua vì trung tính —
       zone_quality luôn = 0 ở đó, không có gì để so sánh chất lượng).

QUYẾT ĐỊNH THIẾT KẾ: dùng buff_controller=None khi eval — TLangReward đã tự
hỗ trợ 2 chế độ (train: buff_controller!=None -> reward += buff; eval:
buff_controller=None -> reward = gate_score + zone_quality, bỏ hẳn buff).
Giống triết lý scripts/eval_val.py bản v1 (đo outcome thật, không lẫn
reward-shaping của train).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM

from app.config.schema import AppConfig
from app.tokenizer.hub import load_tokenizer
from app.training.reward.stats_collector import StatsCollector
from app.training.reward.tlang_reward import TLangReward

ZONE_TYPES = ("NO_ZONE", "SUP_ZONE", "RES_ZONE")
REWARD_RELEVANT_ZONE_TYPES = ("SUP_ZONE", "RES_ZONE")   # NO_ZONE trung tính, bỏ qua ở mean_reward

def print_zone_quality_histogram(
    stats_collector,
    zone_score_weight: float,
    rr_max: int,
    zone_types: Sequence[str] = ("SUP_ZONE", "RES_ZONE"),
) -> None:
    """
    Bucket zone_quality (đã nhân weight, đúng thang với mean_reward) VỀ LẠI
    số R nguyên gần nhất (0R, 1R, 2R, ..., rr_maxR) bằng cách chia ngược
    cho zone_score_weight rồi round — group theo R để đọc trực quan, KHÔNG
    group theo điểm zone_quality trần trụi (0.1, 0.2... khó hình dung hơn
    "1R, 2R...").
    """
    if zone_score_weight <= 0:
        print("zone_score_weight <= 0 — không thể quy đổi ngược về R, bỏ qua histogram.")
        return

    per_type_counts = {zt: defaultdict(int) for zt in zone_types}
    per_type_total = {zt: 0 for zt in zone_types}

    for r in stats_collector._records:
        if r.zone_type not in zone_types or r.zone_quality is None:
            continue
        r_multiple_approx = round(r.zone_quality / zone_score_weight)
        r_multiple_approx = max(0, min(rr_max, r_multiple_approx))   # kẹp về [0, rr_max] phòng sai số round
        per_type_counts[r.zone_type][r_multiple_approx] += 1
        per_type_total[r.zone_type] += 1

    print("\n=== Phân phối zone_quality theo bội số R (SUP/RES) ===")
    for zt in zone_types:
        total = per_type_total[zt]
        print(f"\n{zt} (n={total}):")
        if total == 0:
            print("  (không có sample nào)")
            continue
        for r_level in range(0, rr_max + 1):
            n = per_type_counts[zt].get(r_level, 0)
            ratio = n / total
            bar = "#" * int(ratio * 40)
            print(f"  {r_level:>2}R  count={n:<6} ratio={ratio * 100:5.1f}%  {bar}")

        n_at_cap = per_type_counts[zt].get(rr_max, 0)
        print(f"  -> tỉ lệ chạm cap ({rr_max}R): {n_at_cap / total * 100:.1f}%")
        n_zero = per_type_counts[zt].get(0, 0)
        print(f"  -> tỉ lệ zone_quality=0 (chạm SL gần như ngay): {n_zero / total * 100:.1f}%")

class ZoneEval:
    """
    model_repo: checkpoint task1 (HF Hub repo_id).
    revision: git revision/tag/commit trên Hub — dùng khi cần pin đúng 1
        lần push cụ thể (KHÁC subfolder — 1 repo có thể có nhiều revision
        VÀ nhiều subfolder cùng lúc, tuỳ cách bạn tổ chức checkpoint).
    subfolder: thư mục con trong repo — dùng khi checkpoint nằm trong
        subfolder kiểu "last-checkpoint" (hub_strategy="checkpoint" của
        train_grpo.py push vào đây, KHÁC bản "final" ở root repo).
    dataset_repo: dataset GRPO (schema prompt/future_bins/symbol/window_id),
        val split.
    """

    def __init__(
        self,
        cfg: AppConfig,
        model_repo: str,
        dataset_repo: str,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        split: str = "val",
        tokenizer_repo: Optional[str] = None,
        batch_size: int = 16,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.95,
        limit: Optional[int] = None,
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Tokenizer: pin cùng revision với model nếu có, giữ quirk
        # add_eos_token=False/add_bos_token=True/padding_side=left (batch generate). ---
        self.tok = load_tokenizer(repo_id=tokenizer_repo or model_repo, revision=revision, allow_local_fallback=False)
        self.tok.add_eos_token = False
        self.tok.add_bos_token = True
        self.tok.padding_side = "left"

        model_kwargs: Dict[str, Any] = {}
        if revision is not None:
            model_kwargs["revision"] = revision
        if subfolder is not None:
            model_kwargs["subfolder"] = subfolder
        self.model = LlamaForCausalLM.from_pretrained(model_repo, **model_kwargs).to(self.device)
        self.model.eval()
        if self.model.config.vocab_size != self.tok.vocab_size:
            raise ValueError(
                f"model.vocab_size ({self.model.config.vocab_size}) != tokenizer.vocab_size "
                f"({self.tok.vocab_size}) — checkpoint và tokenizer không khớp "
                f"(model_repo={model_repo!r}, revision={revision!r}, subfolder={subfolder!r})."
            )

        self.dataset = load_dataset(dataset_repo, split=split)
        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        # --- Eval mode: buff_controller=None -> TLangReward tự bỏ hẳn phần
        # buff (reward = gate_score + zone_quality), KHÔNG cần dựng
        # EMABuffController giả lập = 0 nữa (TLangReward đã tự xử lý case
        # này qua tham số buff_controller Optional). ---
        self.stats_collector = StatsCollector()
        self.reward_fn = TLangReward(cfg, buff_controller=None, stats_collector=self.stats_collector)

        # Lưu song song reward TRẢ VỀ THẬT của compute_reward() theo đúng thứ tự
        # log — KHÔNG suy ngược từ hằng số gate_score=2.0 (giả định nội bộ của
        # TLangReward có thể đổi sau này, tránh phụ thuộc ngầm vào con số đó).
        self._rewards: List[float] = []

    # ------------------------------------------------------------------
    # Inference — y hệt ZoneInference._generate_batch (khác chỗ do_sample
    # mặc định False, dùng greedy cho eval reproducible).
    # ------------------------------------------------------------------
    def _generate_batch(self, rows: Sequence[Dict[str, Any]]) -> List[str]:
        prompts = [r["prompt"] for r in rows]
        enc = self.tok(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        gen_ids = out_ids[:, input_ids.shape[1]:]
        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)

    def run(self) -> Dict[str, Any]:
        n = len(self.dataset)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            rows = [self.dataset[i] for i in range(start, end)]
            completions = self._generate_batch(rows)

            for row, completion in zip(rows, completions):
                reward = self.reward_fn.compute_reward(row["prompt"], completion, row["future_bins"])
                self._rewards.append(reward)

            print(f"  ... {end}/{n}")

        return self.summarize()

    # ------------------------------------------------------------------
    # Thống kê — 2 thành phần theo đúng yêu cầu.
    # ------------------------------------------------------------------
    def summarize(self) -> Dict[str, Any]:
        records = self.stats_collector._records
        n = len(records)

        zone_counts, zone_total = self.stats_collector.full_history_counts(key_fn=lambda r: r.zone_type)
        zone_type_ratio = {
            zt: (zone_counts.get(zt, 0) / zone_total if zone_total else 0.0) for zt in ZONE_TYPES
        }

        mean_reward: Dict[str, Optional[float]] = {}
        for zt in REWARD_RELEVANT_ZONE_TYPES:
            rewards = [
                self._rewards[i] for i, r in enumerate(records)
                if r.well_formed and r.semantic_passed and r.zone_type == zt
            ]
            mean_reward[zt] = (sum(rewards) / len(rewards)) if rewards else None

        n_wf = sum(1 for r in records if r.well_formed)
        n_sem = sum(1 for r in records if r.well_formed and r.semantic_passed)

        return {
            "n_samples": n,
            "well_form_rate": (n_wf / n) if n else 0.0,
            "semantic_pass_rate_given_well_formed": (n_sem / n_wf) if n_wf else 0.0,
            "zone_type_ratio": zone_type_ratio,
            "mean_reward": mean_reward,
        }

    def print_summary(self) -> None:
        result = self.summarize()
        print("\n=== ZoneEval summary ===")
        print(f"n_samples = {result['n_samples']}")
        print(f"well_form_rate = {result['well_form_rate'] * 100:.1f}%")
        print(f"semantic_pass_rate (trong số well-formed) = {result['semantic_pass_rate_given_well_formed'] * 100:.1f}%")

        print("\n-- Tỉ lệ zone_type (trong số đã pass gate) --")
        for zt in ZONE_TYPES:
            ratio = result["zone_type_ratio"][zt]
            print(f"  {zt:<10} ratio={ratio * 100:5.1f}%")

        print("\n-- Mean reward theo zone_type (SUP/RES — NO_ZONE bỏ qua vì trung tính) --")
        for zt in REWARD_RELEVANT_ZONE_TYPES:
            mr = result["mean_reward"][zt]
            mr_str = f"{mr:.4f}" if mr is not None else "-"
            print(f"  {zt:<10} mean_reward={mr_str}")
            
        print_zone_quality_histogram(self.stats_collector, self.cfg.base.zone_score_weight, self.cfg.base.rr_max)