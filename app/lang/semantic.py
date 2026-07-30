from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.lang.ast_nodes import ProgramNode, ThinkNode


# =====================================================================
# SemanticResult — passed CHỈ true khi KHÔNG có vi phạm nào (100%, theo
# quyết định đã chốt: gate 2 yêu cầu pass toàn bộ mới cho phép tính
# outcome, không dùng ngưỡng %). `score` vẫn liên tục (dùng cho nhánh
# fail, R_sem_fail) để reward không quá thưa.
# =====================================================================
@dataclass
class SemanticResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    score: float = 1.0


class SemanticChecker:
    VIOLATION_PENALTY = 0.2       # placeholder — tinh chỉnh sau khi có dữ liệu GRPO thực nghiệm

    def __init__(
        self,
        zone_width_min_bins: int,
        zone_width_max_bins: int,
    ) -> None:
        self.zone_width_min_bins = zone_width_min_bins
        self.zone_width_max_bins = zone_width_max_bins

    def check(self, program: ProgramNode) -> SemanticResult:
        chart, think = program.chart, program.think
        violations: List[str] = []

        # Phòng vệ: thiếu thành phần cơ bản để đánh giá — lẽ ra đã bị
        # well-form chặn từ trước (Semantic Checker chỉ nên chạy khi
        # well-form đã pass), nhưng vẫn xử lý an toàn nếu bị gọi độc lập.
        if chart is None or think is None:
            return SemanticResult(passed=False, violations=["Thiếu chart/think/action — không thể kiểm tra semantic"], score=0.0)
        if not chart.candles or think.trend is None or think.current_price_bin is None:
            return SemanticResult(
                passed=False,
                violations=["Thiếu trend/current_price/action_type/candles — không thể kiểm tra semantic"],
                score=0.0,
            )

        self._check_trend_zone(think, violations)
        self._check_zone_direction_vs_price(think, violations)
        self._check_zone_width(think, violations)

        passed = len(violations) == 0
        score = max(0.0, 1.0 - self.VIOLATION_PENALTY * len(violations))
        return SemanticResult(passed=passed, violations=violations, score=score)

    # ------------------------------------------------------------------
    # A. Trend ↔ Zone
    # ------------------------------------------------------------------
    def _check_trend_zone(self, think: ThinkNode, violations: List[str]) -> None:
        trend = think.trend
        zone = think.zone

        if trend == "UP":
            if zone is None:
                violations.append("trend=UP nhưng thiếu zone (bắt buộc phải có zone_support)")
            elif zone.direction != "support":
                violations.append(f"trend=UP nhưng zone lại là {zone.direction} (phải là zone_support)")

        elif trend == "DOWN":
            if zone is None:
                violations.append("trend=DOWN nhưng thiếu zone (bắt buộc phải có zone_resistance)")
            elif zone.direction != "resistance":
                violations.append(f"trend=DOWN nhưng zone lại là {zone.direction} (phải là zone_resistance)")

        elif trend == "RANGE":
            # RANGE: zone tùy chọn, cả 2 hướng đều hợp lệ nếu có — không có vi phạm ở mục A.
            pass

    # ------------------------------------------------------------------
    # B. Hướng của Zone ↔ current_price (bin arithmetic thuần túy)
    # ------------------------------------------------------------------
    def _check_zone_direction_vs_price(self, think: ThinkNode, violations: List[str]) -> None:
        zone = think.zone
        if zone is None:
            return
        current = think.current_price_bin

        if zone.direction == "support":
            if not (zone.lower_bin <= current):
                violations.append(
                    f"zone_support ({zone.lower_bin}:{zone.upper_bin}) nằm hoàn toàn trên current_price "
                    f"({current}) — zone_support phải nằm dưới hoặc chứa giá hiện tại"
                )
        else:  # resistance
            if not (zone.upper_bin >= current):
                violations.append(
                    f"zone_resistance ({zone.lower_bin}:{zone.upper_bin}) nằm hoàn toàn dưới current_price "
                    f"({current}) — zone_resistance phải nằm trên hoặc chứa giá hiện tại"
                )

    # ------------------------------------------------------------------
    # B2. Bề rộng Zone — dùng zone_width_min_bins/zone_width_max_bins truyền
    # từ constructor (T-05: tham số hoá, KHÔNG còn hằng số lớp hardoded).
    # ------------------------------------------------------------------
    def _check_zone_width(self, think: ThinkNode, violations: List[str]) -> None:
        zone = think.zone
        if zone is None:
            return
        width = zone.upper_bin - zone.lower_bin
        if not (self.zone_width_min_bins <= width <= self.zone_width_max_bins):
            violations.append(
                f"zone={zone.direction} ({zone.lower_bin}:{zone.upper_bin}) có width={width} bin, "
                f"ngoài phạm vi hợp lệ [{self.zone_width_min_bins},{self.zone_width_max_bins}]"
            )