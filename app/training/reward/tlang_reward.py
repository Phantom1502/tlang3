from __future__ import annotations

from app.training.reward.zone_buff_controller import EMABuffController

"""
reward.py — Reward function cho GRPO round của TASK1 (zone-inference model),
project tách riêng khỏi task2 (buy/hold/sell).

THIẾT KẾ TỔNG QUAN (nháp, viết ra để nhớ lúc quay lại code tiếp):

    Grammar task1 CHỈ còn chart_block + think_block (trend, current_price,
    zone) — KHÔNG có action/SL/RR (đã xác nhận ở app/lang review trước đó).
    Vì vậy reward ở đây CHỈ có 1 "task" duy nhất (không cần tách zone/action
    như reward_func_v2.py bản v1) — bản này là bản RÚT GỌN của thiết kế v2,
    bỏ hẳn action_task + action_buff_controller + rr_entropy_controller
    (những cái đó thuộc project task2 riêng).

    2 tầng, giống tinh thần v1/v2 (xem app/training/reward/reward_func_v2.py
    cũ để đối chiếu, dù bản đó có thêm action task không áp dụng ở đây):

    Tầng 1 — Common gate (ĐÃ XONG, xem common_check()):
        well_formed (Parser) AND semantic_passed (SemanticChecker: A/B/B2).
        Fail bất kỳ cái nào -> trả thẳng gate_score (điểm liên tục, KHÔNG
        raw 0/1) — reward thưa dần theo số lỗi, không phải nhị phân.

    Tầng 2 — Zone quality (ĐANG LÀM DỞ, xem zone_score()):
        Nếu pass gate: đo "nếu đặt lệnh NGAY TẠI biên zone (upper cho
        support/lower cho resistance) thì đi được bao xa thuận lợi trước khi
        bị SL" — probe_zone_quality(). Đây là tín hiệu "zone có đáng để
        task2 sau này vào lệnh khi giá chạm hay không", KHÔNG phải reward
        cho 1 lệnh thật (task1 không sinh action).

    CÒN THIẾU (xem TODO rải trong code bên dưới, tổng hợp lại đây cho dễ
    nhìn tổng thể trước khi đi vào chi tiết từng hàm):

        [ ] TODO-1 (BUG, ưu tiên cao nhất — xem docstring measure_max_favorable_r):
            probe_zone_quality() gọi measure_max_favorable_r() THIẾU
            outcome_horizon/cap -> TypeError ngay khi có zone. Phải quyết
            định nguồn 2 giá trị này (cfg.window.outcome_horizon là ứng viên
            rõ ràng cho outcome_horizon; cap thì task1 không có RR nên
            KHÔNG thể tái dùng ENTRY_QUALITY_CAP=RR_MAX như v1 — cần 1 hằng
            số/field RoundConfig riêng, ví dụ zone_quality_cap trong round
            config, hoặc 1 constant cố định kiểu ZONE_QUALITY_CAP=hardcode).

        [ ] TODO-2: RoundConfig cho task1 — CHƯA TỒN TẠI. Cần tối thiểu:
            - zone_width_min_bins/zone_width_max_bins (đã có trong
              AppConfig.base, có thể dùng thẳng KHÔNG cần round riêng, vì
              task1 không có "nới/siết zone theo round" như task2 — xác
              nhận lại: có cần round-over-round đổi zone_width không, hay
              zone_width cố định suốt task1 training?).
            - zone_quality_cap (xem TODO-1).
            - buff cho 2 nhóm HAS_ZONE / NO_ZONE (xem TODO-3) — target_ratio,
              buff_min/max/init, PD params (kp/kd/step_max/ema_alpha) —
              y hệt cấu trúc zone namespace trong round_config_v2.py cũ,
              chỉ bỏ hẳn phần action.

        [ ] TODO-3: Buff controller cho tỉ lệ HAS_ZONE/NO_ZONE — CHƯA VIẾT.
            RANGE-không-zone vẫn là 1 output hợp lệ (giống ý nghĩa HOLD ở
            v1) — cần 1 EMA+PD controller y hệt EMABuffControllerV2 (namespace
            "zone", 2 group) để giữ tỉ lệ HAS_ZONE/NO_ZONE không suy biến
            (model đổ xô sinh toàn RANGE-không-zone vì dễ pass gate hơn, hay
            ngược lại). Có thể tái dùng thẳng
            app/training/reward/buff_controller_v2.py (đã viết, tổng quát
            theo namespace) thay vì viết lại — CHỈ cần define GROUPS_ZONE =
            ("HAS_ZONE", "NO_ZONE") và RoundConfig tương ứng.

        [ ] TODO-4: compute_reward() — thân hàm sau common gate hiện
            `raise NotImplementedError`. Cần:
                zone_task = self.zone_score(program, future_bins)
                buffed = zone_task.zone_quality + buff_controller.get_buff(
                    "HAS_ZONE" if zone_task.has_zone else "NO_ZONE"
                )
                reward = common_result.gate_score + buffed
                buff_controller.record(...)  # nếu dùng kiểu record-based như v1,
                                              # hoặc dùng counts-based như v2
                                              # (StatsCollectorV2.counts_since_step_boundary)
                -- CHỌN 1 TRONG 2 KIỂU ĐẾM (record() nội bộ như v1 EMABuffController,
                hay counts truyền ngoài như v2 EMABuffControllerV2) — xem lại
                2 file cũ để chọn, ĐỪNG trộn 2 kiểu.

        [ ] TODO-5: RolloutRecord/StatsCollector (hoặc tương đương) — CHƯA CÓ.
            Cần tối thiểu để report được: trend, has_zone, well_formed,
            semantic_passed, zone_quality, reward — dùng để debug/theo dõi
            tỉ lệ HAS_ZONE/NO_ZONE + phân phối zone_quality qua các step,
            giống StatsCollectorV2.print_summary() bản v1 (bỏ phần action).

        [ ] TODO-6: __call__() hiện chưa log gì vào StatsCollector, chưa gọi
            buff_controller.record()/on_step_end() ở đâu cả (on_step_end nên
            nằm ở TrainerCallback bên train_grpo.py, giống pattern v1 —
            KHÔNG gọi trong __call__ vì __call__ chạy nhiều lần/step, còn
            on_step_end chỉ nên chạy đúng 1 lần/step).

        [ ] TODO-7 (nhỏ, để ý sau): common_check() khi passed=True luôn trả
            gate_score = semantic_result.score + parse_result.well_form_score()
            — nhưng passed=True chỉ xảy ra khi CẢ 2 đều pass (0 lỗi), nghĩa
            là 2 score này LUÔN đúng bằng 1.0 mỗi cái -> gate_score khi pass
            LUÔN đúng bằng 2.0 cố định, không cần cộng runtime. Có thể thay
            bằng hằng số (rõ ràng hơn, đỡ tính lại vô ích) hoặc giữ nguyên
            cho code tự-document — không phải bug, chỉ là 1 cách viết khác.
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from enum import Enum

from app.config.schema import AppConfig, RoundConfig
from app.config.loader import get_round_config
from app.data_prepare.candle import Candle
from app.lang.ast_nodes import ProgramNode, ZoneNode
from app.lang.parser import Parser, ParseResult
from app.lang.semantic import SemanticChecker, SemanticResult
from app.training.reward.stats_collector import StatsCollector, TaskRolloutMeta

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

    TODO-1 (BUG — xem chi tiết ở docstring module): mọi lời gọi hàm này
    PHẢI truyền đủ outcome_horizon/cap. Hiện `probe_zone_quality()` gọi
    THIẾU 2 tham số này -> TypeError ngay khi có zone. Chưa fix ở đây vì
    cần quyết định nguồn của 2 giá trị (outcome_horizon chắc chắn lấy từ
    cfg.window.outcome_horizon; cap thì task1 không còn RR nên không thể
    tái dùng RR_MAX như v1 — cần 1 hằng số/field config riêng cho "trần R
    hợp lý khi đo zone quality", ví dụ 5.0 hoặc 1 field RoundConfig mới).
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


def probe_zone_quality(
    zone: ZoneNode,
    future_candles: List[Candle],
    outcome_horizon: int,
    cap: float,
) -> ForwardTestResult:
    """
    Mô phỏng "vào lệnh ngay khi giá chạm mép zone (entry), cắt lỗ ngay khi
    giá phá thủng mép đối diện + buffer" — support: entry=upper_bin,
    sl=lower_bin-buffer, long. resistance: entry=lower_bin, sl=upper_bin+
    buffer, short. Trả r_multiple = max-favorable-R đã đạt trước khi bị SL
    (KHÔNG phải outcome thật, chỉ là thước đo "zone này có đáng chú ý").
    """
    if zone.direction == "support":
        entry, sl, direction = zone.upper_bin, zone.lower_bin - ZONE_PROBE_SL_BUFFER_BINS, "long"
    else:
        entry, sl, direction = zone.lower_bin, zone.upper_bin + ZONE_PROBE_SL_BUFFER_BINS, "short"

    target = measure_max_favorable_r(
        entry, 
        sl, 
        future_candles, 
        direction,
        outcome_horizon=outcome_horizon,
        cap=cap
    )

    return ForwardTestResult(status=OutcomeStatus.WIN, r_multiple=target)


class TLangReward:
    """
    Reward function cho GRPO round của task1 — dùng làm `reward_funcs` cho
    GRPOTrainer (trl), qua __call__(prompts, completions, future_bins, ...).

    TODO-3/TODO-5: __init__ hiện chỉ giữ self.cfg — CẦN THÊM:
        - self.buff_controller: EMABuffControllerV2(groups=("HAS_ZONE","NO_ZONE"), namespace="zone")
          (tái dùng app/training/reward/buff_controller_v2.py, không viết lại).
          Phải seed_from_round_config() hoặc load() từ checkpoint TRƯỚC khi
          train (làm ở train_grpo.py, giống pattern v1 — KHÔNG tự seed
          trong __init__ ở đây vì __init__ không biết đang resume hay round mới).
        - self.stats: StatsCollector-kiểu-rút-gọn (chỉ cần trend/has_zone/
          zone_quality/reward, xem TODO-5) — optional param truyền vào
          __call__ hoặc giữ làm attribute, CHỌN 1 TRONG 2 (đồng nhất với
          cách buff_controller được truyền/giữ).
    """

    def __init__(self, cfg: AppConfig, round_id: str, buff_file_path: Optional[str] = None):
        self.cfg = cfg
        self.round_config = get_round_config(cfg, round_id)
        self.stats_collector = StatsCollector()
        
        # init buff_controller
        groups = ()
        for zone_type, _ in self.round_config.zone_buffs.items():
            groups += (zone_type,)
            
        self.buff_controller = EMABuffController(
            groups=groups, namespace="zone"
        )
        if buff_file_path is not None and Path(buff_file_path).exists():
            self.buff_controller.load(buff_file_path)
        else:
            self.buff_controller.init(self.round_config)
        
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

        future_candles: List[Candle] = [Candle(*b) for b in future_bins]
        probe: ForwardTestResult = probe_zone_quality(
            think.zone, 
            future_candles,
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
        zone_quality = probe.r_multiple * self.round_config.zone_score_weight
        return ZoneTaskScore(
            zone_quality=zone_quality,
            probe=probe,
            has_zone=True,
        )

    def compute_reward(self, prompt: Any, completion: str, future_bins: Sequence[Sequence[int]], **kwargs) -> float:
        """
        TODO-4: thân hàm sau common gate hiện `raise NotImplementedError`.
        Việc còn lại (xem chi tiết đầy đủ ở docstring module, mục TODO-4):
            3. reward = common_result.gate_score + zone_task.zone_quality + buff.
            4. Log vào stats (TODO-5) + record cho buff_controller (TODO-4,
               chọn kiểu record()-nội-bộ hay counts-truyền-ngoài, xem note
               trong TODO-4 ở docstring module — ĐỪNG trộn 2 kiểu).
        """
        reward = 0.0
        
        parse_result: ParseResult = Parser.from_text(prompt + " " + completion).parse()
        program = parse_result.ast
        common_result: CommonGateResult = self.common_check(parse_result, program)
        reward += common_result.gate_score
        if not common_result.passed:
            meta = TaskRolloutMeta(
                trend=program.think.trend,
                well_formed=parse_result.is_well_formed(),
                semantic_passed=False,
                zone_type=None,
                zone_quality=None,
                buff_applied=None
            )
            self.stats_collector.log(meta)
            return common_result.gate_score
        
        
        zone_score: ZoneTaskScore = self.zone_score(program, future_bins)
        # TODO: phần này hơi hard code, nếu ko khớp với config type thì sẽ bị lỗi, nếu có giải pháp khác thì nên thay đổi
        zone_type = self._get_zone_type(program)
        buff = self.buff_controller.get_buff(zone_type)
        reward = reward + zone_score.zone_quality + buff

        meta = TaskRolloutMeta(
            trend=program.think.trend,
            well_formed=True,
            semantic_passed=True,
            zone_type=zone_type,
            zone_quality=zone_score.zone_quality,
            buff_applied=buff
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
            reward = self.compute_reward(prompt, completion, future_bin, **kwargs)
            rewards.append(reward)
        return rewards