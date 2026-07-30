"""
app/config/schema.py

Trach nhiem: Dinh nghia kieu du lieu (dataclass) cho tung nhom tham so cau hinh.
KHONG doc file, KHONG co I/O -- thuan kieu du lieu + validate invariant noi tai.

Pham vi task T-01 (xem 03_task_breakdown/task_breakdown.md):
    BaseConfig, WindowConfig, ScaleEntry, ModelPreset, ModelsConfig, DataGenV2Config

KHONG thuoc pham vi T-01 (se lam o T-02): RoundConfig (mo rong buff 7-action +
SEM_FULL/ACTION_GATE_FULL/weights) va AppConfig (gop tat ca lai). Hai kieu nay
CHUA duoc dinh nghia trong file nay -- de tranh lan pham vi sang T-02, theo dung
contract da dong bang o interfaces.md.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any


@dataclass(frozen=True)
class BaseConfig:
    bin_min: int
    bin_max: int
    n_bins: int
    zone_width_min_bins: int
    zone_width_max_bins: int
    digit_pad: int
    rr_min: int
    rr_max: int
    action_types: Tuple[str, ...]  # 7 gia tri co dinh theo PRD 4.1.c
    trend_values: Tuple[str, ...]  # UP/DOWN/RANGE


@dataclass(frozen=True)
class WindowConfig:
    input_candles: int      # 100 o v2
    outcome_horizon: int    # 100 o v2
    window_size: int        # PHAI = input_candles + outcome_horizon

    def __post_init__(self) -> None:
        if self.window_size != self.input_candles + self.outcome_horizon:
            raise ValueError(
                "WindowConfig.window_size phai bang input_candles + outcome_horizon "
                f"(nhan duoc window_size={self.window_size}, "
                f"input_candles={self.input_candles}, "
                f"outcome_horizon={self.outcome_horizon})"
            )


@dataclass(frozen=True)
class ScaleEntry:
    symbol: str
    timeframe: str
    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"ScaleEntry.scale phai > 0 (nhan duoc {self.scale})")


@dataclass(frozen=True)
class ModelPreset:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int


@dataclass(frozen=True)
class ModelsConfig:
    vocab_size: int
    max_position_embeddings: int
    presets: Dict[str, ModelPreset] = field(default_factory=dict)  # key: "tiny"/"small"/"base"/"large"


@dataclass(frozen=True)
class DataGenV2Config:
    stride: int                 # buoc nhay cua so truot -- PHAI < window_size
    n_augments_per_window: int  # so ban augment sinh them moi cua so that

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError(f"DataGenV2Config.stride phai > 0 (nhan duoc {self.stride})")
        if self.n_augments_per_window < 0:
            raise ValueError(
                "DataGenV2Config.n_augments_per_window phai >= 0 "
                f"(nhan duoc {self.n_augments_per_window})"
            )
         
@dataclass(frozen=True)
class TrainingConfig:
    phase: str
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    logging_steps: int
    save_steps: int
    
    def __post_init__(self):
        # Validate & convert learning_rate
        try:
            lr_val = float(self.learning_rate)
            object.__setattr__(self, "learning_rate", lr_val)
        except (ValueError, TypeError):
            raise TypeError(f"[{self.phase}] learning_rate không thể convert sang float: {self.learning_rate!r}")

        # Validate các field int
        int_fields = ["batch_size", "gradient_accumulation_steps", "warmup_steps", "max_steps", "logging_steps", "save_steps"]
        for field in int_fields:
            val = getattr(self, field)
            try:
                object.__setattr__(self, field, int(val))
            except (ValueError, TypeError):
                raise TypeError(f"[{self.phase}] {field} không thể convert sang int: {val!r}")
            
# --- T-02: RoundConfig (mo rong) + AppConfig -------------------------------------------------

# Best-case sem_score khi FAIL Common gate = dung 1 violation trong toan bo A/B/B2/D/E,
# dung lai penalty scheme san co (VIOLATION_PENALTY=0.2) => max(0, 1 - 0.2) = 0.8.
# Nguon: reward_design_v2.md muc 4 ("Tinh cu the SEM_FULL").
VIOLATION_PENALTY_MAX: float = 0.8

# 7 action_type rieng biet cho buff v2 (KHONG gop nhom nhu v1) -- reward_design_v2.md muc 5.
REQUIRED_ACTION_TYPES_V2: Tuple[str, ...] = (
    "HOLD", "BUY", "SELL", "CANCEL_BUY", "CANCEL_SELL", "WAIT_BUY", "WAIT_SELL",
)


@dataclass(frozen=True)
class ActionBuffConfig:
    """Cau hinh buff cho DUNG 1 action_type (khong gop nhom nhu v1)."""
    buff_min: float
    buff_max: float
    buff_init: float
    target_ratio: float

    def __post_init__(self) -> None:
        if self.buff_min > self.buff_max:
            raise ValueError(...)
        if not (self.buff_min <= self.buff_init <= self.buff_max):
            raise ValueError(...)


@dataclass(frozen=True)
class RoundConfig:
    round_id: str
    zone_width_min_bins: int
    zone_width_max_bins: int
    sl_min_dist_bins: int
    sl_max_dist_bins: int
    trade_fee_bins: float
    # --- reward v2 ---
    SEM_FULL: float
    ACTION_GATE_FULL: float
    zone_score_weight: float
    entry_score_weight: float
    action_buffs: Dict[str, ActionBuffConfig]  # DU 7 muc, key trong REQUIRED_ACTION_TYPES_V2

    def __post_init__(self) -> None:
        # Invariant 1 (muc 4): worst PASS Common gate > best FAIL Common gate
        if self.SEM_FULL <= VIOLATION_PENALTY_MAX:
            raise ValueError(...)
        # Invariant 2 (muc 5): worst PASS gate rieng task action > best FAIL gate rieng
        if self.sl_min_dist_bins <= 0:
            raise ValueError(...)
        fee_worst = self.trade_fee_bins / self.sl_min_dist_bins
        if self.ACTION_GATE_FULL <= 1.0 + fee_worst:
            raise ValueError(...)
        # R3: buff phai tach rieng DU 7 action, khong gop nhom
        missing = set(REQUIRED_ACTION_TYPES_V2) - set(self.action_buffs.keys())
        if missing:
            raise ValueError(...)


@dataclass(frozen=True)
class AppConfig:
    base: BaseConfig
    window: WindowConfig
    scales: List[ScaleEntry]
    models: ModelsConfig
    training_defaults: TrainingConfig
    datagen_v2: DataGenV2Config
    rounds: Dict[str, RoundConfig]