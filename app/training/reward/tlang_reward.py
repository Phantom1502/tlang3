from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence
from enum import Enum

from app.config.schema import AppConfig, RoundConfig
from app.data_prepare.candle import Candle
from app.lang.ast_nodes import ProgramNode, ZoneNode
from app.lang.parser import Parser, ParseResult
from app.lang.semantic import SemanticChecker, SemanticResult
from app.training.reward.stats_collector import StatsCollector, TaskRolloutMeta
from app.training.reward.zone_buff_controller import EMABuffController

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
    """Trạng thái forward-test — task1 hiện chỉ dùng WIN (probe luôn "thắng"
    theo nghĩa đo max-favorable-R, không có SL thật để LOSE) và
    INVALID_SETUP (risk=0, zone suy biến). LOSS/TIMEOUT giữ lại cho tương lai
    nếu sau này task1 cần forward-test kiểu nhị phân WIN/LOSS/TIMEOUT thật
    (hiện chưa cần, vì task1 không có action/SL/RR để "đóng lệnh")."""
    WIN = "WIN"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"
    INVALID_SETUP = "INVALID_SETUP"


@dataclass
class ForwardTestResult:
    status: OutcomeStatus
    r_multiple: float
    exit_index: Optional[int] = None


@dataclass
class ZoneTaskScore:
    zone_quality: float          # r_multiple của probe, 0.0 nếu không có zone hoặc INVALID_SETUP
    probe: Optional[ForwardTestResult]
    has_zone: bool

def measure_max_favorable_r(
    entry_bin: int,
    sl_bin: int,
    future_candles: List[Candle],
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

def _find_first_touch(zone: ZoneNode, candles: List[Candle]) -> Optional[int]:
    for i, c in enumerate(candles):
        if c.low <= zone.upper_bin and c.high >= zone.lower_bin:
            return i
    return None

def probe_zone_quality(
    zone: ZoneNode,
    future_candles: List[Candle],
    outcome_horizon: int,
    cap: float,
) -> ForwardTestResult:
    touch_idx = _find_first_touch(zone, future_candles[:outcome_horizon])
    if touch_idx is None:
        # điều kiện #1 KHÔNG thoả — zone không bao giờ được chạm trong horizon
        return ForwardTestResult(status=OutcomeStatus.INVALID_SETUP, r_multiple=0.0)

    if zone.direction == "support":
        entry, sl, direction = zone.upper_bin, zone.lower_bin - ZONE_PROBE_SL_BUFFER_BINS, "long"
    else:
        entry, sl, direction = zone.lower_bin, zone.upper_bin + ZONE_PROBE_SL_BUFFER_BINS, "short"

    remaining_horizon = outcome_horizon - touch_idx
    target = measure_max_favorable_r(
        entry, sl, future_candles[touch_idx:], direction,
        outcome_horizon=remaining_horizon, cap=cap,
    )
    return ForwardTestResult(status=OutcomeStatus.WIN, r_multiple=target)

class TLangReward:
    """
    Reward function cho GRPO round của task1 — dùng làm `reward_funcs` cho
    GRPOTrainer (trl), qua __call__(prompts, completions, future_bins, ...).
    """

    def __init__(
        self,
        cfg: AppConfig,
        round_config: Optional[RoundConfig] = None, # Hiện tại không còn dùng round_config vì zone_score_weight đã chuyển vào cfg.base.zone_score_weight
        buff_controller: Optional[EMABuffController] = None,
        stats_collector: Optional[StatsCollector] = None,
    ):
        """
        Có 2 chế độ chạy cơ bản:
        1. Training: buff_controller != None, stats_collector != None, reward = gate_score + zone_quality + buff
        2. Evaluation: buff_controller = None, stats_collector = None, reward = gate_score + zone_quality

        Args:
            cfg (AppConfig): _description_
            round_config (RoundConfig): _description_
            buff_controller (Optional[EMABuffController], optional): _description_. Defaults to None.
            stats_collector (Optional[StatsCollector], optional): _description_. Defaults to None.
        """
        self.__name__ = "TLangReward"
        self.cfg = cfg
        self.buff_controller = buff_controller
        self.stats_collector = stats_collector
        
    def _get_zone_type(self, program: ProgramNode) -> str:
        if program.think.zone is None:
            return "NO_ZONE"
        if program.think.zone.direction == "support":
            return "SUP_ZONE"
        return "RES_ZONE"

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

        semantic_result: SemanticResult = SemanticChecker(
            zone_width_min_bins=self.cfg.base.zone_width_min_bins,
            zone_width_max_bins=self.cfg.base.zone_width_max_bins,
        ).check(program)
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
        program: ProgramNode,
        future_bins: Sequence[Sequence[int]],
    ) -> ZoneTaskScore:
        """Đo chất lượng zone qua probe_zone_quality(). CHỈ gọi khi
        common_check() đã pass (caller — compute_reward — chịu trách nhiệm
        đảm bảo điều này, hàm này không tự check lại passed)."""
        think = program.think
        if think.zone is None:
            return ZoneTaskScore(
                zone_quality=0.0,
                probe=None,
                has_zone=False,
            )

        last_10_candles_nodes = program.chart.candles[-10:]
        last_10_candles = [Candle(cn.o, cn.h, cn.l, cn.c) for cn in last_10_candles_nodes]
        future_candles: List[Candle] = [Candle(*b) for b in future_bins]
        verify_candles = last_10_candles + future_candles
        probe: ForwardTestResult = probe_zone_quality(
            think.zone, 
            verify_candles,
            outcome_horizon=self.cfg.window.outcome_horizon,
            cap=self.cfg.base.rr_max
        )
        if probe.status == OutcomeStatus.INVALID_SETUP:
            return ZoneTaskScore(
                zone_quality=0.0,
                probe=probe,
                has_zone=True,
            )

        # Apply scale factor, r_multiple in range [0, rr_max] -> zone_quality in range [0, rr_max * zone_score_weight]
        zone_quality = probe.r_multiple * self.cfg.base.zone_score_weight
        return ZoneTaskScore(
            zone_quality=zone_quality,
            probe=probe,
            has_zone=True,
        )

    def compute_reward(self, prompt: Any, completion: str, future_bins: Sequence[Sequence[int]]) -> float:
        reward = 0.0
        
        parse_result: ParseResult = Parser.from_text(self.cfg, prompt + " " + completion).parse()
        program = parse_result.ast
        common_result: CommonGateResult = self.common_check(parse_result, program)
        reward += common_result.gate_score
        if not common_result.passed:
            if self.stats_collector is not None:
                meta = TaskRolloutMeta(
                    trend=program.think.trend if program.think else None,
                    well_formed=parse_result.is_well_formed(),
                    semantic_passed=False,
                    zone_type=None,
                    zone_quality=None,
                    buff_applied=None
                )
                self.stats_collector.log(meta)
            return reward
        
        
        zone_score: ZoneTaskScore = self.zone_score(program, future_bins)
        zone_type = self._get_zone_type(program)
        
        if self.buff_controller is not None:
            buff = self.buff_controller.get_buff(zone_type)
            reward = reward + zone_score.zone_quality + buff
        else:
            reward = reward + zone_score.zone_quality

        if self.stats_collector is not None:
            meta = TaskRolloutMeta(
                trend=program.think.trend if program.think else None,
                well_formed=True,
                semantic_passed=True,
                zone_type=zone_type,
                zone_quality=zone_score.zone_quality,
                buff_applied=buff if self.buff_controller is not None else None
            )
            self.stats_collector.log(meta)
        return reward

    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[str],
        future_bins: Sequence[Sequence[Sequence[int]]],
        **kwargs,
    ) -> List[float]:
        """Entry point cho GRPOTrainer(reward_funcs=...). TODO-6: hiện chưa
        gọi bất kỳ logging/report nào — cần quyết định stats sống ở đâu
        (attribute của self, hay module-level singleton giống
        stats_collector_v2 bản v1) TRƯỚC khi thêm logging vào đây, để
        tránh phải sửa lại 2 lần."""
        rewards = []
        for prompt, completion, future_bin in zip(prompts, completions, future_bins):
            reward = self.compute_reward(prompt, completion, future_bin)
            rewards.append(reward)
        return rewards