from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Dict
from enum import Enum
from collections import defaultdict, Counter

from app.config.schema import AppConfig, RoundConfig
from tlang import (
    CandleNode,
    ProgramNode,
    ZoneNode,
    
    Parser,
    ParseResult,
    SemanticChecker,
    SemanticResult
)
from app.training.reward.stats_collector import StatsCollector, TaskRolloutMeta
from app.training.reward.zone_entropy_controller import ZoneEntropyController, MIN_SAMPLES_PER_GROUP_FOR_ENTROPY

# Số bin đệm giữa SL "giả lập" và mép zone khi probe chất lượng zone — SL đặt
# NGAY SÁT ngoài zone (mép đối diện với entry) để mô phỏng "vào lệnh ngay khi
# giá vừa chạm biên zone, cắt lỗ ngay khi giá phá qua toàn bộ zone".
ZONE_PROBE_SL_BUFFER_BINS = 1


@dataclass
class CommonGateResult:
    """Kết quả gate chung (well-formed + semantic) — DÙNG CHUNG cho mọi
    completion trước khi tính zone_score. `gate_score` là điểm liên tục
    (không nhị phân) để GRPO có gradient mượt ngay cả khi fail gate."""
    program: Optional[ProgramNode]
    well_formed: bool
    semantic_result: Optional[SemanticResult]
    passed: bool                 # well_formed AND semantic_result.passed
    gate_score: float


class OutcomeStatus(Enum):
    """
    WIN: đã CHẠM zone (điều kiện #1 thoả), r_multiple = max-favorable-R đo
        được TỪ ĐÚNG thời điểm chạm (KHÔNG PHẢI từ nến đầu tiên của window).
    ZONE_NOT_TOUCHED: zone hợp lệ về hình học/hướng nhưng giá KHÔNG BAO GIỜ
        chạm trong suốt outcome_horizon — r_multiple = 0.0 (KHÔNG dùng số
        khác 0 để đánh dấu case này, vì r_multiple là thang đo dùng để TÍNH
        REWARD, không phải chỗ nhét cờ trạng thái — trạng thái đã có field
        `status`/`is_touched` riêng để phân biệt).
    INVALID_SETUP: risk=0 (entry==sl, zone suy biến) — GIỮ LẠI trong enum
        cho tương lai/phòng thủ, nhưng với công thức entry/sl hiện tại
        (entry=mép zone, sl=mép đối diện-buffer) risk KHÔNG THỂ = 0 với
        zone_width>=zone_width_min_bins>0 — probe_zone_quality() hiện KHÔNG
        BAO GIỜ trả status này, xem zone_score() không còn check nhánh này.
    LOSS/TIMEOUT: chưa dùng — giữ lại cho tương lai nếu task1 cần
        forward-test nhị phân WIN/LOSS/TIMEOUT thật (task1 không có
        action/SL/RR để "đóng lệnh" nên hiện chưa cần).
    """
    WIN = "WIN"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"
    INVALID_SETUP = "INVALID_SETUP"
    ZONE_NOT_TOUCHED = "ZONE_NOT_TOUCHED"


@dataclass
class ForwardTestResult:
    status: OutcomeStatus
    r_multiple: float
    exit_index: Optional[int] = None


@dataclass
class ZoneTaskScore:
    zone_quality: float            # r_multiple đã nhân zone_score_weight, 0.0 nếu không có zone HOẶC không chạm
    probe: Optional[ForwardTestResult]
    has_zone: bool
    is_touched: Optional[bool]     # None nếu không có zone (NO_ZONE) — KHÔNG dùng False cho case này


def measure_max_favorable_r(
    entry_bin: int,
    sl_bin: int,
    future_candles: List[CandleNode],
    direction: str,
    outcome_horizon: int,
    cap: float,
) -> float:
    """
    Đo R thuận lợi lớn nhất đã đạt được trước khi chạm sl_bin/hết
    outcome_horizon/chạm trần cap — hàm THUẦN TÚY, không phụ thuộc
    self.cfg (test/gọi độc lập dễ dàng).
    """
    risk = abs(entry_bin - sl_bin)
    if risk == 0:
        return 0.0

    max_r = 0.0
    for candle in future_candles[:outcome_horizon]:
        if direction == "long":
            if candle.low <= sl_bin:
                break
            max_r = max(max_r, (candle.high - entry_bin) / risk)
        else:
            if candle.high >= sl_bin:
                break
            max_r = max(max_r, (entry_bin - candle.low) / risk)
        if max_r >= cap:
            max_r = cap
            break
    return max_r


def _find_first_touch(zone: ZoneNode, candles: List[CandleNode]) -> Optional[int]:
    """Index nến ĐẦU TIÊN có [low,high] giao với [zone.lower_bin,
    zone.upper_bin] — None nếu không nến nào chạm trong toàn bộ `candles`
    (caller đã cắt đúng outcome_horizon trước khi truyền vào)."""
    for i, c in enumerate(candles):
        if c.low <= zone.upper_bin and c.high >= zone.lower_bin:
            return i
    return None


def probe_zone_quality(
    zone: ZoneNode,
    future_candles: List[CandleNode],
    outcome_horizon: int,
    cap: float,
) -> ForwardTestResult:
    """
    Mô phỏng "vào lệnh NGAY KHI giá chạm mép zone (entry), cắt lỗ ngay khi
    giá phá thủng mép đối diện + buffer" — support: entry=upper_bin,
    sl=lower_bin-buffer, long. resistance: entry=lower_bin, sl=upper_bin+
    buffer, short.

    BẮT BUỘC điều kiện #1 (zone phải được giá tương lai CHẠM TỚI) trước khi
    tính điều kiện #2 (không bị SL) — nếu không chạm, trả ZONE_NOT_TOUCHED
    NGAY, KHÔNG giả định "vào lệnh từ nến đầu tiên của window" như bản cũ
    (bản cũ để lọt qua case zone đặt xa giá hiện tại vẫn được điểm max nếu
    trend tự nhiên đủ mạnh — không phản ánh đúng "model chọn zone khéo").
    """
    touch_idx = _find_first_touch(zone, future_candles[:outcome_horizon])
    if touch_idx is None:
        return ForwardTestResult(status=OutcomeStatus.ZONE_NOT_TOUCHED, r_multiple=0.0)

    if zone.direction == "support":
        entry, sl, direction = zone.upper_bin, zone.lower_bin - ZONE_PROBE_SL_BUFFER_BINS, "long"
    else:
        entry, sl, direction = zone.lower_bin, zone.upper_bin + ZONE_PROBE_SL_BUFFER_BINS, "short"

    remaining_horizon = outcome_horizon - touch_idx
    target = measure_max_favorable_r(
        entry, 
        sl, 
        future_candles[touch_idx:], 
        direction,
        outcome_horizon=remaining_horizon, 
        cap=cap,
    )
    return ForwardTestResult(status=OutcomeStatus.WIN, r_multiple=target)


class TLangReward:
    """
    Reward function cho GRPO round của task1 — dùng làm `reward_funcs` cho
    GRPOTrainer (trl), qua __call__(prompts, completions, future_bins, ...).

    2 chế độ chạy:
        1. Training: buff_controller != None, stats_collector != None ->
           reward = gate_score + zone_quality + buff.
        2. Evaluation: buff_controller = None -> reward = gate_score +
           zone_quality (bỏ hẳn buff). stats_collector vẫn nên truyền vào
           (dù None cũng chạy được, chỉ là không log gì) để đọc lại
           TaskRolloutMeta (bao gồm is_touched) sau khi eval.
    """

    def __init__(
        self,
        cfg: AppConfig,
        entropy_controller: Optional[ZoneEntropyController] = None,
        entropy_position_controller: Optional[ZoneEntropyController] = None,
        stats_collector: Optional[StatsCollector] = None,
    ):
        self.__name__ = "TLangReward"
        self.cfg = cfg
        self.entropy_controller = entropy_controller
        self.entropy_position_controller = entropy_position_controller
        self.stats_collector = stats_collector

    def common_check(
        self,
        parse_result: ParseResult,
        program: ProgramNode,
    ) -> CommonGateResult:
        if not parse_result.is_well_formed():
            return CommonGateResult(
                program=program,
                well_formed=False,
                semantic_result=None,
                passed=False,
                gate_score=parse_result.well_form_score(),
            )

        semantic_result: SemanticResult = SemanticChecker(self.cfg.tlang_zone).check(program)
        if not semantic_result.passed:
            return CommonGateResult(
                program=program,
                well_formed=True,
                semantic_result=semantic_result,
                passed=False,
                gate_score=semantic_result.score,
            )

        return CommonGateResult(
            program=program,
            well_formed=True,
            semantic_result=semantic_result,
            passed=True,
            gate_score=semantic_result.score + parse_result.well_form_score(),
        )

    def zone_score(
        self,
        zone: ZoneNode,
        future_bins: List[CandleNode],
    ) -> ZoneTaskScore:
        """Đo chất lượng zone qua probe_zone_quality(). CHỈ gọi khi
        common_check() đã pass (caller — compute_reward — chịu trách nhiệm
        đảm bảo điều này, hàm này không tự check lại passed)."""
        if zone is None:
            return ZoneTaskScore(
                zone_quality=self.cfg.base.no_zone_reward * self.cfg.base.zone_score_weight, 
                probe=None, 
                has_zone=False, 
                is_touched=None
            )

        probe: ForwardTestResult = probe_zone_quality(
            zone,
            future_bins,
            outcome_horizon=self.cfg.window.outcome_horizon,
            cap=self.cfg.base.rr_max,
        )

        if probe.status == OutcomeStatus.ZONE_NOT_TOUCHED:
            # KHÔNG cộng/trừ gì — chỉ ghi nhận trạng thái để quan sát qua
            # StatsCollector (xem touch_rate_by_zone_type()). Nếu quan sát
            # thấy tỉ lệ not-touched tăng bất thường qua các round, quay
            # lại bàn thêm penalty RIÊNG BIỆT, KHÔNG lẫn vào zone_quality.
            return ZoneTaskScore(zone_quality=0.0, probe=probe, has_zone=True, is_touched=False)
        
        return ZoneTaskScore(
            zone_quality=probe.r_multiple * self.cfg.base.zone_score_weight, 
            probe=probe, 
            has_zone=True, 
            is_touched=True
        )

    def compute_reward(self, prompt: Any, completion: str, future_bins: List[CandleNode]) -> Tuple[float, TaskRolloutMeta]:
        reward = 0.0

        parse_result: ParseResult = Parser.from_text(self.cfg, prompt + " " + completion).parse()
        program = parse_result.ast
        common_result: CommonGateResult = self.common_check(parse_result, program)
        reward += common_result.gate_score
        if not common_result.passed:
            meta = TaskRolloutMeta(
                trend=program.think.trend if program.think else None,
                well_formed=parse_result.is_well_formed(),
                semantic_passed=False,
                zone_type=None,
                zone_quality=None,
                zone_upper=None,
                zone_lower=None,
                is_touched=None,
            )
            if self.stats_collector is not None:
                self.stats_collector.log(meta)
            return reward, meta

        zone_score: ZoneTaskScore = self.zone_score(program.think.zone, future_bins)
        reward = reward + zone_score.zone_quality

        meta = TaskRolloutMeta(
            trend=program.think.trend if program.think else None,
            well_formed=True,
            semantic_passed=True,
            zone_type=program.think.zone_type,
            zone_quality=zone_score.zone_quality,
            zone_upper=program.think.zone_upper,
            zone_lower=program.think.zone_lower,
            is_touched=zone_score.is_touched,
        )
        if self.stats_collector is not None:
            self.stats_collector.log(meta)
        return reward, meta

    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[str],
        future_bins: Sequence[Sequence[Sequence[int]]],
        **kwargs,
    ) -> List[float]:
        """Entry point cho GRPOTrainer(reward_funcs=...)."""
        n = len(prompts)

        rewards: List[float] = [0.0] * n
        metas: List[Optional[TaskRolloutMeta]] = [None] * n
        
        for i in range(n):
            future_candles: List[CandleNode] = [CandleNode(*b) for b in future_bins[i]]
            reward, meta = self.compute_reward(
                prompts[i], 
                completions[i], 
                future_candles
            )
            rewards[i] = reward
            metas[i] = meta
        
        groups_idx: Dict[Any, List[int]] = defaultdict(list)
        for i, prompt in enumerate(prompts):
            if metas[i].well_formed and metas[i].semantic_passed:
                groups_idx[prompt].append(i)

        strength = self.entropy_controller.get_bonus()
        for idx_list in groups_idx.values():
            if len(idx_list) < MIN_SAMPLES_PER_GROUP_FOR_ENTROPY:
                continue

            branch_list = [f"{metas[i].trend}|{metas[i].zone_type}" for i in idx_list]
            h, probs = _entropy_and_probs_str(branch_list)
            self.entropy_controller.record_entropy(h)

            if strength <= 0.0:
                continue

            for i in idx_list:
                branch_key = f"{metas[i].trend}|{metas[i].zone_type}"
                surprisal = -math.log(probs[branch_key])
                max_suprisal = -math.log(1.0 / n)  # surprisal trần khi p=1/16 (hiếm nhất có thể trong group 16)
                normalized_surprisal = surprisal / max_suprisal
                rewards[i] += strength * normalized_surprisal
                
        pos_strength = self.entropy_position_controller.get_bonus()
        for idx_list in groups_idx.values():
            if len(idx_list) < MIN_SAMPLES_PER_GROUP_FOR_ENTROPY:
                continue

            pos_branch_list = [f"{metas[i].zone_upper}|{metas[i].zone_lower}" for i in idx_list]
            h, probs = _entropy_and_probs_str(pos_branch_list)
            self.entropy_position_controller.record_entropy(h)

            if pos_strength <= 0.0:
                continue

            for i in idx_list:
                pos_branch_key = f"{metas[i].zone_upper}|{metas[i].zone_lower}"
                surprisal = -math.log(probs[pos_branch_key])
                max_suprisal = -math.log(1.0 / n)
                normalized_surprisal = surprisal / max_suprisal
                rewards[i] += pos_strength * normalized_surprisal
                
        return rewards
    
def _entropy_and_probs_str(values: Sequence[str]) -> Tuple[float, Dict[str, float]]:
    n = len(values)
    counts = Counter(values)
    probs = {v: c / n for v, c in counts.items()}
    h = -sum(p * math.log(p) for p in probs.values())
    return h, probs