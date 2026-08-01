from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Sequence
from app.config.schema import RoundConfig, GroupBuffState
from app.config.loader import get_buff_group

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class EMABuffController:
    def __init__(self, groups: Sequence[str], namespace: str):
        self.groups = tuple(groups)
        self.namespace = namespace
        self.states: Dict[str, GroupBuffState] = {}
        
    def init(self, round_config: RoundConfig):
        for group in self.groups:
            self.states[group] = get_buff_group(round_config, group)
            
    def on_step_end(self, round_config: RoundConfig, counts: Dict[str, int], total: int) -> None:
        if total <= 0:
            return
        alpha = round_config.alpha
        kp = round_config.kp
        kd = round_config.kd
        step_max = round_config.step_max
        
        for group in self.groups:
            state = self.states[group]
            count = counts.get(group, 0)
            ratio = count / total
            
            # EMA update
            ema_ratio = alpha * ratio + (1 - alpha) * state.ema_ratio
            
            # Error and derivative
            error = ema_ratio - round_config.zone_buffs[group].target_ratio
            d_error = error - state.prev_error
            
            # Buff update
            buff_delta = kp * error + kd * d_error
            buff_delta = _clip(buff_delta, -step_max, step_max)
            new_buff = _clip(state.buff + buff_delta, round_config.zone_buffs[group].buff_min, round_config.zone_buffs[group].buff_max)
            
            # Update state
            self.states[group] = GroupBuffState(
                ema_ratio=ema_ratio,
                buff=new_buff,
                prev_error=error
            )
    
    def get_buff(self, group: str) -> float:
        state = self.states.get(group)
        if state is None:
            return 0.0
        return state.buff
    
    def snapshot(self) -> Dict[str, GroupBuffState]:
        return {
            g: {"ema_ratio": s.ema_ratio, "buff": s.buff, "prev_error": s.prev_error}
            for g, s in self.states.items()
        }
        
    def state_dict(self) -> Dict[str, Dict[str, float]]:
        return self.snapshot()
    
    def load_state_dict(self, data: Dict[str, Dict[str, float]]) -> None:
        for group, d in data.items():
            self.states[group] = GroupBuffState(
                ema_ratio=float(d["ema_ratio"]),
                buff=float(d["buff"]),
                prev_error=float(d.get("prev_error", 0.0)),
            )
    
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), ensure_ascii=False), encoding="utf-8")
        
    def load(self, path: str) -> bool:
        """Trả True nếu load thành công. Caller PHẢI gọi
        seed_from_round_config() khi trả về False — KHÔNG được để states
        rỗng (get_buff sẽ âm thầm trả 0.0 cho group thiếu)."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.load_state_dict(data)
            return True
        except Exception:
            return False
        
    @classmethod
    def load_or_init(cls, round_config: RoundConfig, resume_checkpoint: str = None) -> EMABuffController:
        groups = tuple(round_config.zone_buffs.keys())
        buff_controller = EMABuffController(groups=groups, namespace="zone")

        import os
        buff_path = os.path.join(resume_checkpoint, "zone_buff_state.json") if resume_checkpoint else None
        if buff_path and Path(buff_path).exists():
            buff_controller.load(buff_path)
        else:
            buff_controller.init(round_config)   # round MỚI hoặc load thất bại -> seed lại từ config

        return buff_controller