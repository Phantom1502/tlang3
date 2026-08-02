"""
demos/reward_demo.py — Demo TLangReward (task1, zone-inference GRPO reward).

Dựng vài completion mẫu (well-form fail / semantic fail / pass+no-zone /
pass+support+zone tốt / pass+resistance+zone tệ) rồi in breakdown từng
thành phần (gate_score, zone_quality, buff, tổng) — để "nhìn thấy" điểm số
trước khi cắm vào GRPO thật.

Chạy: python -m demos.reward_demo
"""
from app.config.schema import (
    AppConfig, BaseConfig, ModelsConfig, RoundConfig, WindowConfig, ZoneBuffConfig,
)
from app.training.reward.stats_collector import StatsCollector
from app.training.reward.tlang_reward import TLangReward
from app.training.reward.zone_buff_controller import EMABuffController

N_CANDLES = 5   # nhỏ để test dễ đọc — KHÔNG liên quan configs/window.yaml thật

base_cfg = BaseConfig(
    bin_min=0, bin_max=1023, n_bins=1024,
    zone_width_min_bins=5, zone_width_max_bins=20,
    digit_pad=4, rr_min=1, rr_max=9,
    action_types=("BUY", "SELL", "HOLD"),
    trend_values=("UP", "DOWN", "RANGE"),
)
window_cfg = WindowConfig(input_candles=N_CANDLES, outcome_horizon=N_CANDLES, window_size=2 * N_CANDLES)
models_cfg = ModelsConfig(vocab_size=100, max_position_embeddings=512, presets={})

# round1.yaml thật (zone_score_weight/alpha/kp/kd/step_max + 3 nhóm zone_buffs)
round_config = RoundConfig(
    round_id="round1",
    zone_score_weight=0.2,
    alpha=0.1, kp=0.1, kd=0.1, step_max=100,
    zone_buffs={
        "NO_ZONE": ZoneBuffConfig(buff_min=-0.5, buff_max=0.5, buff_init=0.0, target_ratio=0.2),
        "SUP_ZONE": ZoneBuffConfig(buff_min=-0.5, buff_max=0.5, buff_init=0.0, target_ratio=0.4),
        "RES_ZONE": ZoneBuffConfig(buff_min=-0.5, buff_max=0.5, buff_init=0.0, target_ratio=0.4),
    },
)

cfg = AppConfig(
    base=base_cfg, window=window_cfg, scales=[], models=models_cfg,
    training_defaults=[], rounds={round_config.round_id: round_config},
)


def fmt_bin(n: int, pad: int = None) -> str:
    pad = pad if pad is not None else cfg.base.digit_pad
    return " ".join(str(n).zfill(pad))


def make_chart(closes) -> str:
    candles = [f"<O_{c}> <H_{c + 5}> <L_{c - 5}> <C_{c}>" for c in closes]
    return "<chart> " + " ".join(candles) + " </chart>"


def make_think(trend: str, current_price: int, zone=None) -> str:
    text = f"<think> <trend>{trend}</trend> <current_price> {fmt_bin(current_price)} </current_price>"
    if zone is not None:
        direction, lower, upper = zone
        tag = "zone_support" if direction == "support" else "zone_resistance"
        text += f" <{tag}> {fmt_bin(lower)} : {fmt_bin(upper)} </{tag}>"
    text += " </think>"
    return text


def flat_future(n: int, o: int, h: int, l: int, c: int):
    return [[o, h, l, c]] * n


def run_case(reward_fn: TLangReward, name: str, prompt: str, completion: str, future_bins) -> None:
    reward = reward_fn.compute_reward(prompt, completion, future_bins)
    meta = reward_fn.stats_collector._records[-1]

    print(f"\n=== {name} ===")
    print(f"  well_formed     = {meta.well_formed}")
    print(f"  semantic_passed = {meta.semantic_passed}")
    if meta.semantic_passed:
        gate_score = reward - meta.zone_quality - meta.buff_applied
        print(f"  gate_score      = {gate_score:.4f}  (cố định 2.0 khi pass cả 2 gate)")
        print(f"  zone_type       = {meta.zone_type}")
        print(f"  zone_quality    = {meta.zone_quality:.4f}  (đã nhân zone_score_weight={round_config.zone_score_weight})")
        print(f"  buff            = {meta.buff_applied:.4f}")
    else:
        print(f"  gate_score      = {reward:.4f}  (fail gate -> reward = gate_score, không có zone/buff)")
    print(f"  reward TỔNG     = {reward:.4f}")


def run() -> None:
    buff_controller = EMABuffController(groups=tuple(round_config.zone_buffs.keys()), namespace="zone")
    buff_controller.init(round_config)   # buff = buff_init = 0.0 cho cả 3 nhóm lúc mới seed

    reward_fn = TLangReward(cfg, round_config, buff_controller, StatsCollector())

    # ------------------------------------------------------------
    # Case 1: well-form fail (garbage hoàn toàn) -> gate_score thấp, không có zone/buff
    # ------------------------------------------------------------
    run_case(
        reward_fn, "garbage (well-form fail)",
        prompt="", completion="hoan toan khong theo grammar", future_bins=flat_future(N_CANDLES, 0, 0, 0, 0),
    )

    # ------------------------------------------------------------
    # Case 2: well-form pass, semantic fail (trend UP nhưng thiếu zone -> rule A)
    # ------------------------------------------------------------
    closes = [500, 505, 503, 507, 512]
    run_case(
        reward_fn, "UP thiếu zone (semantic fail, rule A)",
        prompt=make_chart(closes), completion=make_think("UP", 512, zone=None),
        future_bins=flat_future(N_CANDLES, 500, 505, 495, 500),
    )

    # ------------------------------------------------------------
    # Case 3: pass cả 2 gate, RANGE không zone -> NO_ZONE, zone_quality=0.0
    # ------------------------------------------------------------
    closes = [500, 505, 503, 507, 500]
    run_case(
        reward_fn, "RANGE không zone (pass gate, NO_ZONE)",
        prompt=make_chart(closes), completion=make_think("RANGE", 500, zone=None),
        future_bins=flat_future(N_CANDLES, 500, 505, 495, 500),
    )

    # ------------------------------------------------------------
    # Case 4: pass gate, UP + zone_support TỐT — giá đi thuận lợi nhiều trước
    # khi (giả sử) chạm SL. entry=upper_bin=510, sl=lower_bin-1=499, risk=11.
    # Future toàn thắng, không nến nào chạm sl=499 -> max_r lớn.
    # ------------------------------------------------------------
    closes = [500, 505, 503, 507, 510]
    future_good = [
        [510, 520, 505, 515],
        [515, 530, 510, 525],
        [525, 545, 520, 540],
        [540, 560, 535, 555],
        [555, 570, 550, 565],
    ]
    run_case(
        reward_fn, "UP + zone_support TỐT (giá chạy thuận lợi, không chạm SL)",
        prompt=make_chart(closes), completion=make_think("UP", 510, zone=("support", 500, 510)),
        future_bins=future_good,
    )

    # ------------------------------------------------------------
    # Case 5: pass gate, DOWN + zone_resistance TỆ — giá đảo chiều gần như
    # ngay lập tức. entry=lower_bin=600, sl=upper_bin+1=613, risk=13.
    # Nến đầu tiên đã chạm sl (h=615>=613) -> max_r=0.0 (chưa kịp ghi nhận
    # thuận lợi nào trước khi "thua").
    # ------------------------------------------------------------
    closes = [590, 595, 593, 597, 600]
    future_bad = [[600, 615, 598, 610]] + flat_future(N_CANDLES - 1, 605, 608, 600, 605)
    run_case(
        reward_fn, "DOWN + zone_resistance TỆ (đảo chiều ngay, chạm SL sớm)",
        prompt=make_chart(closes), completion=make_think("DOWN", 600, zone=("resistance", 600, 612)),
        future_bins=future_bad,
    )

    print("\n" + "=" * 70)
    print("So sánh: zone TỐT (case 4) nên có reward CAO HƠN HẲN zone TỆ (case 5)")
    print("và cả 2 đều CAO HƠN case fail-gate (case 1, 2) — kiểm tra bằng mắt ở trên.")
    print("=" * 70)

    reward_fn.stats_collector.print_summary()


if __name__ == "__main__":
    run()