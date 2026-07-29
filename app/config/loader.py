"""
app/config/loader.py

Trach nhiem: Doc toan bo config/*.yaml (ke ca config/rounds/*.yaml), validate, tra ve
DUNG 1 AppConfig da rap day du. DIEM NAP CAU HINH DUY NHAT cua toan he thong (K1-K3) --
khong module nao khac duoc tu doc file YAML.

Contract: interfaces.md § Module: app/config/loader.py
"""

import os
from typing import Any, Dict

import yaml

from app.config.schema import (
    AppConfig,
    BaseConfig,
    WindowConfig,
    ScaleEntry,
    ModelPreset,
    ModelsConfig,
    DataGenV2Config,
    RoundConfig,
    ActionBuffConfig,
)

# Cac file bat buoc phai co truc tiep trong config_dir (khong ke rounds/, duoc xu ly rieng).
_REQUIRED_TOP_LEVEL_FILES = (
    "base.yaml",
    "window.yaml",
    "scales.yaml",
    "models.yaml",
    "training_defaults.yaml",
    "datagen_v2.yaml",
)
_ROUNDS_SUBDIR = "rounds"


def _read_yaml(path: str) -> Any:
    """Doc 1 file YAML, tra ve du lieu da parse (dict/list/...)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_field(data: Dict[str, Any], key: str, source: str) -> Any:
    """Lay 1 field bat buoc tu dict; raise ValueError neu thieu (khong fallback am tham)."""
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"Thieu field bat buoc '{key}' trong {source}")
    return data[key]


def _build_base_config(data: Dict[str, Any], source: str) -> BaseConfig:
    return BaseConfig(
        bin_min=_require_field(data, "bin_min", source),
        bin_max=_require_field(data, "bin_max", source),
        digit_pad=_require_field(data, "digit_pad", source),
        rr_min=_require_field(data, "rr_min", source),
        rr_max=_require_field(data, "rr_max", source),
        action_types=tuple(_require_field(data, "action_types", source)),
        trend_values=tuple(_require_field(data, "trend_values", source)),
    )


def _build_window_config(data: Dict[str, Any], source: str) -> WindowConfig:
    return WindowConfig(
        input_candles=_require_field(data, "input_candles", source),
        outcome_horizon=_require_field(data, "outcome_horizon", source),
        window_size=_require_field(data, "window_size", source),
    )


def _build_scale_entries(data: Any, source: str) -> list:
    if not isinstance(data, list):
        raise ValueError(f"{source} phai la 1 danh sach cac scale entry")
    entries = []
    for i, item in enumerate(data):
        item_source = f"{source}[{i}]"
        entries.append(
            ScaleEntry(
                symbol=_require_field(item, "symbol", item_source),
                timeframe=_require_field(item, "timeframe", item_source),
                window_size=_require_field(item, "window_size", item_source),
                scale=_require_field(item, "scale", item_source),
            )
        )
    return entries


def _build_models_config(data: Dict[str, Any], source: str) -> ModelsConfig:
    presets_raw = data.get("presets", {}) or {}
    presets = {}
    for name, preset_data in presets_raw.items():
        preset_source = f"{source}.presets.{name}"
        presets[name] = ModelPreset(
            hidden_size=_require_field(preset_data, "hidden_size", preset_source),
            num_hidden_layers=_require_field(preset_data, "num_hidden_layers", preset_source),
            num_attention_heads=_require_field(preset_data, "num_attention_heads", preset_source),
            num_key_value_heads=_require_field(preset_data, "num_key_value_heads", preset_source),
            intermediate_size=_require_field(preset_data, "intermediate_size", preset_source),
        )
    return ModelsConfig(
        tokenizer_repo=_require_field(data, "tokenizer_repo", source),
        max_position_embeddings=_require_field(data, "max_position_embeddings", source),
        presets=presets,
    )


def _build_datagen_v2_config(data: Dict[str, Any], source: str) -> DataGenV2Config:
    return DataGenV2Config(
        stride=_require_field(data, "stride", source),
        n_augments_per_window=_require_field(data, "n_augments_per_window", source),
    )


def _build_round_config(data: Dict[str, Any], source: str) -> RoundConfig:
    action_buffs_raw = _require_field(data, "action_buffs", source)
    action_buffs = {}
    for action_type, buff_data in action_buffs_raw.items():
        buff_source = f"{source}.action_buffs.{action_type}"
        action_buffs[action_type] = ActionBuffConfig(
            buff_min=_require_field(buff_data, "buff_min", buff_source),
            buff_max=_require_field(buff_data, "buff_max", buff_source),
            buff_init=_require_field(buff_data, "buff_init", buff_source),
            target_ratio=_require_field(buff_data, "target_ratio", buff_source),
        )
    return RoundConfig(
        round_id=_require_field(data, "round_id", source),
        zone_width_min_bins=_require_field(data, "zone_width_min_bins", source),
        zone_width_max_bins=_require_field(data, "zone_width_max_bins", source),
        sl_min_dist_bins=_require_field(data, "sl_min_dist_bins", source),
        sl_max_dist_bins=_require_field(data, "sl_max_dist_bins", source),
        trade_fee_bins=_require_field(data, "trade_fee_bins", source),
        SEM_FULL=_require_field(data, "SEM_FULL", source),
        ACTION_GATE_FULL=_require_field(data, "ACTION_GATE_FULL", source),
        zone_score_weight=_require_field(data, "zone_score_weight", source),
        entry_score_weight=_require_field(data, "entry_score_weight", source),
        action_buffs=action_buffs,
    )


def load_config(config_dir: str = "./config") -> AppConfig:
    """
    Pre-condition: config_dir ton tai, chua du file bat buoc (base.yaml, window.yaml,
        scales.yaml, models.yaml, training_defaults.yaml, datagen_v2.yaml, rounds/*.yaml).
    Post-condition: tra ve dung 1 AppConfig da validate (moi __post_init__ cua
        schema.py deu pass).
    Raises:
        FileNotFoundError -- thieu 1 trong cac file bat buoc.
        ValueError -- 1 file co field bat buoc bi thieu, hoac gia tri vi pham
            invariant dinh nghia trong app/config/schema.py.
    Side-effect: chi doc file, KHONG ghi, KHONG mutate global state nao.
    """
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"config_dir khong ton tai: {config_dir}")

    paths = {}
    for filename in _REQUIRED_TOP_LEVEL_FILES:
        path = os.path.join(config_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Thieu file cau hinh bat buoc: {path}")
        paths[filename] = path

    rounds_dir = os.path.join(config_dir, _ROUNDS_SUBDIR)
    if not os.path.isdir(rounds_dir):
        raise FileNotFoundError(f"Thieu thu muc cau hinh bat buoc: {rounds_dir}")
    round_files = sorted(
        f for f in os.listdir(rounds_dir) if f.endswith(".yaml") or f.endswith(".yml")
    )
    if not round_files:
        raise FileNotFoundError(f"Thu muc {rounds_dir} khong chua file round nao (*.yaml)")

    base_data = _read_yaml(paths["base.yaml"])
    window_data = _read_yaml(paths["window.yaml"])
    scales_data = _read_yaml(paths["scales.yaml"])
    models_data = _read_yaml(paths["models.yaml"])
    training_defaults_data = _read_yaml(paths["training_defaults.yaml"])
    datagen_v2_data = _read_yaml(paths["datagen_v2.yaml"])

    base = _build_base_config(base_data, paths["base.yaml"])
    window = _build_window_config(window_data, paths["window.yaml"])
    scales = _build_scale_entries(scales_data, paths["scales.yaml"])
    models = _build_models_config(models_data, paths["models.yaml"])
    datagen_v2 = _build_datagen_v2_config(datagen_v2_data, paths["datagen_v2.yaml"])

    if not isinstance(training_defaults_data, dict):
        raise ValueError(f"{paths['training_defaults.yaml']} phai la 1 mapping (dict)")

    rounds: Dict[str, RoundConfig] = {}
    for round_filename in round_files:
        round_path = os.path.join(rounds_dir, round_filename)
        round_data = _read_yaml(round_path)
        round_config = _build_round_config(round_data, round_path)
        rounds[round_config.round_id] = round_config

    return AppConfig(
        base=base,
        window=window,
        scales=scales,
        models=models,
        training_defaults=training_defaults_data,
        datagen_v2=datagen_v2,
        rounds=rounds,
    )


def get_scale(config: AppConfig, symbol: str, timeframe: str, window_size: int) -> float:
    """
    Pre-condition: config da load thanh cong.
    Post-condition: tra ve dung scale factor khop CHINH XAC (symbol, timeframe, window_size).
    Raises: KeyError neu khong co entry khop -- KHONG fallback am tham ve gia tri mac dinh.
    """
    for entry in config.scales:
        if (
            entry.symbol == symbol
            and entry.timeframe == timeframe
            and entry.window_size == window_size
        ):
            return entry.scale
    raise KeyError(
        f"Khong tim thay ScaleEntry khop (symbol={symbol!r}, timeframe={timeframe!r}, "
        f"window_size={window_size!r})"
    )


def get_round_config(config: AppConfig, round_id: str) -> RoundConfig:
    """
    Pre-condition: config da load thanh cong.
    Post-condition: tra ve dung RoundConfig khop round_id.
    Raises: KeyError neu round_id khong ton tai.
    """
    if round_id not in config.rounds:
        raise KeyError(f"Khong tim thay RoundConfig cho round_id={round_id!r}")
    return config.rounds[round_id]