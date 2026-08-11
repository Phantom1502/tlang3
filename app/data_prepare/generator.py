from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.candle import Candle
from app.lang import (
    ASTVisitor,
    ThinkNode,
    ZoneNode,
    SemanticChecker,
    Parser
)
from app.config import (
    AppConfig,
    BaseConfig,
)

@dataclass
class GeneratedSample:
    prompt: str        # "<chart>...</chart>"
    completion: str     # "<think>...</think><action>...</action>"
    leaf_recipe: str     # tên leaf-path đã dùng — để kiểm tra phân phối (mục A, lượt trước)

# =====================================================================
# Cây leaf-path hợp lệ — LIỆT KÊ TƯỜNG MINH trước khi sample, đúng
# nguyên tắc "sample uniform trên toàn bộ leaf-path hợp lệ, không sample
# từng field độc lập rồi lọc bỏ invalid" (spec mục 7.2).
#
# Mỗi leaf = (trend, zone_side, zone_case, action_type)
#   zone_side: "support" | "resistance" | None (RANGE không zone)
#   zone_case: "CONTAINS" | "TOUCH" | "NOTOUCH" | None
#     CONTAINS -> current_price nằm trong zone -> price_in_zone bắt buộc True
#     TOUCH    -> zone nằm ngoài current_price nhưng 1 trong 5 nến cuối chạm -> price_in_zone=True
#     NOTOUCH  -> zone nằm ngoài, không nến nào chạm -> price_in_zone=False
# =====================================================================
LEAF_RECIPES: List[Tuple[str, Optional[str]]] = [
    # trend=UP — chỉ zone_support, action phía buy
    ("UP", "support"),
    # trend=DOWN — chỉ zone_resistance, action phía sell
    ("DOWN", "resistance"),
    # trend=RANGE — có thể có zone_support HOẶC zone_resistance HOẶC không zone
    ("RANGE", "support"),
    ("RANGE", "resistance"),
    ("RANGE", None),
]

class ZoneGenerator:
    def __init__(self, cfg: AppConfig, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        base_cfg: BaseConfig = cfg.base
        
        self.zone_min = base_cfg.zone_width_min_bins
        self.zone_max = base_cfg.zone_width_max_bins
        self.bin_min = base_cfg.bin_min
        self.bin_max = base_cfg.bin_max
        
        self._random = random.Random(seed)
        self._visitor = ASTVisitor(digit_pad=cfg.base.digit_pad)
        
    def _pick_zone(
        self, 
        side: str,
        current_price: int,
    ) -> Optional[ZoneNode]:
        width = self._random.randint(self.zone_min, self.zone_max)
        
        if side == "support":
            lower = self._random.randint(0, current_price)
            upper = lower + width
            if lower < self.bin_min or upper > self.bin_max:
                return None
            return ZoneNode(direction="support", lower_bin=lower, upper_bin=upper)
        elif side == "resistance":
            upper = self._random.randint(current_price, self.bin_max)
            lower = upper - width
            if lower < self.bin_min or upper > self.bin_max:
                return None
            return ZoneNode(direction="resistance", lower_bin=lower, upper_bin=upper)
        
        return None

    def generate_one(
        self, 
        candles: List[Candle],
        max_attempts: int = 30,
    ) -> Optional[GeneratedSample]:
        current_price = candles[-1].close
        
        for _ in range(max_attempts):
            trend, side = self._random.choice(LEAF_RECIPES)
            think = ThinkNode(trend=trend, current_price_bin=current_price)
            
            if side is None:
                # RANGE không zone -> HOLD
                pass
            else:
                zone = self._pick_zone(side, current_price)
                if zone is None:
                    continue
                
                think.zone = zone
        
            completion = self._visitor.build_completion(think)
            prompt = self._visitor.render_chart_block(candles)
            
            full_text = prompt + " " + completion
            parse_result = Parser.from_text(self.cfg,full_text).parse()
            if not parse_result.is_well_formed():
                continue
            
            sem_result = SemanticChecker(self.zone_min, self.zone_max).check(parse_result.ast)
            if not sem_result.passed:
                continue
                    
            leaf_name = f"{trend}|{side}"
            return GeneratedSample(prompt, completion, leaf_name)
        
        return None
    
    def generate_dataset(
        self, 
        charts: List[List[Candle]],
        samples_per_chart: int = 4,
        max_attempts: int = 30,
    ) -> List[GeneratedSample]:
        samples: List[GeneratedSample] = []
        for chart in charts:
            for _ in range(samples_per_chart):
                sample = self.generate_one(chart, max_attempts=max_attempts)
                if sample is not None:
                    samples.append(sample)
        return samples