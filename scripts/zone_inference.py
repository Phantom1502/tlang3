"""
app/training/inference/zone_inference.py — Tầng 1 của pipeline self-gen:
chạy batch inference model task1 trên dataset GRPO gốc (schema prompt/
future_bins/symbol/window_id), verify từng completion, ghi kết quả ra
parquet (có resume nếu Colab/Kaggle bị ngắt giữa chừng).

Output schema (1 dòng = 1 completion đã verify):
    prompt, completion, future_bins, symbol, window_id,
    well_formed, semantic_passed, zone_type, zone_quality, price_in_zone_now

LƯU Ý: zone_quality lưu RAW (r_multiple thô, CHƯA nhân zone_score_weight)
— vì trọng số này thuộc RoundConfig, có thể đổi giữa các lần dùng lại
dataset này (build_task1_retrain_rows ở tầng sau tự nhân weight khi cần,
không bake cứng vào đây).

RESUME: tiến trình được coi là 1 chuỗi shard-parquet (mỗi shard <=
shard_size dòng) + 1 file progress.json ghi next_index/shard_idx. Lần
chạy lại đọc progress.json, tiếp tục đúng từ next_index, KHÔNG ghi đè
shard cũ. Không dùng ParquetWriterUtil (context-manager 1 phiên) vì
parquet không hỗ trợ append vào file đã đóng — ghi nhiều file part là
cách đơn giản nhất, load_dataset("parquet", data_files=glob) đọc gộp
nhiều file bình thường.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from app.config import AppConfig
from app.data_prepare import Candle

from app.lang import (
    ZoneNode,
    Parser,
    SemanticChecker
)
from app.training.reward.tlang_reward import OutcomeStatus, probe_zone_quality
from app.inference import ModelInference

# Khớp LAST_N_CANDLES_TOUCH bản v1 cũ (rule D trước khi bị bỏ khỏi
# ThinkNode) — TRÙNG với app/data_prepare/self_gen_dataset_builder.py.
# NẾU đã có file đó trong repo, NÊN import lại từ đó thay vì định nghĩa
# song song ở đây (tránh 2 nơi định nghĩa cùng 1 logic rồi lệch nhau khi
# sửa sau này) — để tạm ở đây vì chưa chắc file kia đã có trong repo.
EXTEND_ZONE_MULTIPLIER = 1

def _is_price_in_zone_now(chart: List[Candle], zone: ZoneNode, last_n: int, extend_multiplier: float = EXTEND_ZONE_MULTIPLIER) -> bool:
    """
    Điều kiện để pass qua action model:
    - trong vòng n nến cuối giá phải chạm zone
    - giá hiện tại ko nằm ngoài zone mở rộng

    Args:
        chart (List[Candle]): _description_
        zone (ZoneNode): _description_
        last_n (int): Số lượng nến cuối cùng để kiểm tra
        extend_multiplier (float): Hệ số mở rộng zone
    Returns:
        bool: _description_
    """
    current_price = chart[-1].close
    extend_zone_range = (zone.upper_bin - zone.lower_bin) * extend_multiplier
    is_current_price_in_extend_zone = (zone.lower_bin - extend_zone_range <= current_price <= zone.upper_bin + extend_zone_range)
    last_n_candles = chart[-last_n:]
    return any(c.low <= zone.upper_bin and c.high >= zone.lower_bin for c in last_n_candles) and is_current_price_in_extend_zone


@dataclass
class ScoreResult:
    well_formed: bool
    semantic_passed: bool
    zone_type: Optional[str]
    zone_quality: Optional[float]   # raw r_multiple, None nếu chưa pass gate
    zone_touched: Optional[bool]    # None nếu không có zone

class ZoneInference:
    """
    model_repo: checkpoint task1 (HF Hub hoặc local dir) dùng để generate.
    dataset_repo: dataset GRPO gốc — HF Hub repo_id, HOẶC path local
        (.parquet file/thư mục) — tự nhận diện qua _load_input_dataset().
    output_dir: nơi ghi shard parquet + progress.json.
    """

    def __init__(
        self,
        cfg: AppConfig,
        model_repo: str,
        dataset_repo: str,
        output_dir: str,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        split: str = "train",
        tokenizer_repo: Optional[str] = None,
        batch_size: int = 16,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        shard_size: int = 2000,
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.shard_size = shard_size

        os.makedirs(output_dir, exist_ok=True)
        self.progress_path = os.path.join(output_dir, "progress.json")

        self.model: ModelInference = ModelInference(
            model_repo, 
            revision=revision, 
            subfolder=subfolder, 
            tokenizer_repo=tokenizer_repo, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=top_p
        )

        self.dataset = self._load_input_dataset(dataset_repo, split)

        self.schema = pa.schema([
            ("prompt", pa.string()),
            ("completion", pa.string()),
            ("future_bins", pa.list_(pa.list_(pa.int16()))),
            ("symbol", pa.string()),
            ("window_id", pa.string()),
            ("well_formed", pa.bool_()),
            ("semantic_passed", pa.bool_()),
            ("zone_type", pa.string()),
            ("zone_quality", pa.float32()),
            ("zone_touched", pa.bool_()),
            ("price_in_zone_now", pa.bool_()),
        ])

        self.next_index, self.shard_idx = self._load_progress()

    # ------------------------------------------------------------------
    # Dataset — tự nhận diện HF Hub repo hay local parquet path.
    # ------------------------------------------------------------------
    @staticmethod
    def _load_input_dataset(dataset_repo: str, split: str):
        if dataset_repo.endswith(".parquet") or os.path.exists(dataset_repo):
            return load_dataset("parquet", data_files=dataset_repo, split="train")
        return load_dataset(dataset_repo, split=split)

    # ------------------------------------------------------------------
    # Progress — resume đúng vị trí dừng trước đó.
    # ------------------------------------------------------------------
    def _load_progress(self) -> tuple:
        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data["next_index"]), int(data["shard_idx"])
        return 0, 0

    def _save_progress(self) -> None:
        with open(self.progress_path, "w", encoding="utf-8") as f:
            json.dump({"next_index": self.next_index, "shard_idx": self.shard_idx}, f)

    # ------------------------------------------------------------------
    # Verify — TÁCH RIÊNG 2 hàm theo đúng yêu cầu: 1 hàm chấm điểm (dùng
    # future_bins), 1 hàm check price-in-zone (KHÔNG dùng future_bins).
    # ------------------------------------------------------------------
    def _score(self, prompt: str, completion: str, future_bins: Sequence[Sequence[int]]):
        """Trả (ScoreResult, program|None). program=None nếu chưa pass gate
        — caller (_check_price_in_zone) cần program để lấy chart+zone."""
        parse_result = Parser.from_text(self.cfg, prompt + " " + completion).parse()
        if not parse_result.is_well_formed():
            return ScoreResult(False, False, None, None, None), None

        program = parse_result.ast
        semantic_result = SemanticChecker(
            zone_width_min_bins=self.cfg.base.zone_width_min_bins,
            zone_width_max_bins=self.cfg.base.zone_width_max_bins,
        ).check(program)
        if not semantic_result.passed:
            return ScoreResult(True, False, None, None, None), None

        zone = program.think.zone
        zone_quality = 0.0
        zone_touched: Optional[bool] = None
        if zone is not None:
            future_candles = [Candle(*b) for b in future_bins]
            probe = probe_zone_quality(
                zone, 
                future_candles,
                outcome_horizon=self.cfg.window.outcome_horizon,
                cap=self.cfg.base.rr_max,
            )
            if probe.status != OutcomeStatus.INVALID_SETUP:
                zone_quality = probe.r_multiple
                zone_touched = (probe.status != OutcomeStatus.ZONE_NOT_TOUCHED)

        return ScoreResult(True, True, program.think.zone_type, zone_quality, zone_touched), program

    def _check_price_in_zone(self, program) -> bool:
        """CHỈ gọi khi program không None và program.think.zone không None
        (caller kiểm tra trước) — KHÔNG dùng future_bins, chỉ dùng chart
        hiện tại (giống rule D bản v1)."""
        chart: List[Candle] = [
            Candle(open=cn.o, high=cn.h, low=cn.l, close=cn.c) for cn in program.chart.candles
        ]
        return _is_price_in_zone_now(chart, program.think.zone, last_n=self.cfg.base.zone_last_n_touch)

    # ------------------------------------------------------------------
    # Ghi parquet — 1 shard/lần flush, KHÔNG append vào file cũ.
    # ------------------------------------------------------------------
    def _flush(self, records: List[dict]) -> None:
        if not records:
            return
        path = os.path.join(self.output_dir, f"inference_part_{self.shard_idx:05d}.parquet")
        table = pa.Table.from_pylist(records, schema=self.schema)
        pq.write_table(table, path)
        print(f"  Đã ghi shard {path} ({len(records)} dòng)")
        self.shard_idx += 1
        self._save_progress()

    # ------------------------------------------------------------------
    # Entry point.
    # ------------------------------------------------------------------
    def run(self) -> None:
        from collections import Counter
        zone_types_counter = Counter()
        
        n = len(self.dataset)
        if self.next_index >= n:
            print(f"Đã xử lý hết {n} dòng từ lần chạy trước — không còn gì để làm.")
            return

        print(f"Bắt đầu từ index {self.next_index}/{n} (shard tiếp theo: {self.shard_idx}).")
        batch_records: List[dict] = []

        while self.next_index < n:
            end = min(self.next_index + self.batch_size, n)
            rows = [self.dataset[i] for i in range(self.next_index, end)]

            completions = self.model.generate_batch(rows)

            for row, completion in zip(rows, completions):
                score, program = self._score(row["prompt"], completion, row["future_bins"])

                price_in_zone_now = False
                if program is not None and program.think.zone is not None:
                    price_in_zone_now = self._check_price_in_zone(program)

                batch_records.append({
                    "prompt": row["prompt"],
                    "completion": completion,
                    "future_bins": row["future_bins"],
                    "symbol": row.get("symbol"),
                    "window_id": row.get("window_id"),
                    "well_formed": score.well_formed,
                    "semantic_passed": score.semantic_passed,
                    "zone_type": score.zone_type,
                    "zone_quality": score.zone_quality,
                    "zone_touched": score.zone_touched,
                    "price_in_zone_now": price_in_zone_now,
                })

                if score.zone_type is not None:
                    zone_types_counter[score.zone_type] += 1

            # debug zone counter
            print(f"Zone type counter: {zone_types_counter}")
            self.next_index = end
            print(f"  ... {self.next_index}/{n}")

            if len(batch_records) >= self.shard_size or self.next_index >= n:
                self._flush(batch_records)
                batch_records = []

        print(f"Hoàn tất — đã xử lý {n} dòng, {self.shard_idx} shard.")