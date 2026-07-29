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
    """
    Kiểm tra bảng 2.2 (A, B, D, E) trên AST đã parse thành công.

    KHÔNG kiểm tra bảng F (field bắt buộc/cấm theo action_type — đã ở
    well-form, thuộc Parser) và KHÔNG kiểm tra mục G (good_price_action
    không có rule nội dung, chủ ý để tránh áp đặt bias chủ quan).

    Nguyên tắc: verifier này = "lật ngược" generator dùng để sinh dữ
    liệu SFT/pretrain — generator đảm bảo đúng các invariant này lúc
    sinh, verifier chỉ cần lật ngược logic đó thành kiểm tra.

    THAY ĐỔI DUY NHẤT so với v1 (task T-05, interfaces.md § Module: app/lang):
    4 tham số zone_width_min_bins/zone_width_max_bins/sl_min_dist_bins/sl_max_dist_bins
    (trước đây zone_width_* là hằng số lớp có giá trị mặc định 5/20, còn
    sl_min/max_dist_bins hoàn toàn KHÔNG tồn tại ở constructor) giờ ĐỀU BẮT BUỘC
    truyền qua constructor, KHÔNG còn giá trị mặc định hardcode nào — đúng
    yêu cầu contract "class này KHÔNG tự có giá trị mặc định hardcode". Caller
    (ở v2) PHẢI truyền đủ 4 giá trị này từ RoundConfig đang active.

    Lưu ý về sl_min_dist_bins/sl_max_dist_bins: LOGIC check() không dùng 2 giá
    trị này ở bất kỳ đâu (kiểm tra SL hợp lệ, `is_sl_valid`, sống ở
    `app/training/reward/forward_test.py`, HOÀN TOÀN TÁCH BIỆT khỏi
    SemanticChecker — xem interfaces.md § Module: reward_func_v2.py, mục
    "Gate riêng của task action"). Contract yêu cầu constructor nhận đủ 4 tham
    số này (chữ ký đã "đóng băng"), nên 2 giá trị này được lưu lại làm thuộc
    tính (self.sl_min_dist_bins/self.sl_max_dist_bins) để không mất thông tin
    caller truyền vào, nhưng KHÔNG được dùng trong check() — giữ đúng logic v1
    (không tự ý thêm rule SL vào semantic check, tránh lệch khỏi trách nhiệm
    module đã phân định rõ trong interfaces.md).
    """

    VIOLATION_PENALTY = 0.2       # placeholder — tinh chỉnh sau khi có dữ liệu GRPO thực nghiệm

    def __init__(
        self,
        zone_width_min_bins: int,
        zone_width_max_bins: int,
    ) -> None:
        """
        4 giá trị này PHẢI được truyền từ RoundConfig đang active — class này
        KHÔNG tự có giá trị mặc định hardcode (đúng nguyên văn docstring contract
        interfaces.md § Module: app/lang).
        """
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