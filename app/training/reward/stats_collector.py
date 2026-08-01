
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Sequence, Any
from collections import defaultdict, Counter

@dataclass
class TaskRolloutMeta:
    trend: Optional[str]
    well_formed: bool
    semantic_passed: bool
    zone_type: Optional[str]         # no_zone / sup_zone / res_zone
    zone_quality: Optional[float]    # = zone_task.zone_quality
    buff_applied: Optional[float]    # = buff (để audit riêng phần buff đóng góp)
    
class StatsCollector:
    def __init__(self) -> None:
        self._records: List[TaskRolloutMeta] = []
        self._step_boundary: int = 0

    def log(self, meta: TaskRolloutMeta) -> None:
        self._records.append(meta)

    def reset(self) -> None:
        self._records.clear()
        self._step_boundary = 0

    def mark_step_boundary(self) -> None:
        self._step_boundary = len(self._records)

    @staticmethod
    def _filter_and_count(records: Sequence[TaskRolloutMeta], key_fn) -> Tuple[Dict[str, int], int]:
        counts: Dict[str, int] = defaultdict(int)
        total = 0
        for r in records:
            if not r.well_formed or not r.semantic_passed:
                continue
            key = key_fn(r)
            if key is None:
                continue
            counts[key] += 1
            total += 1
        return dict(counts), total

    def counts_since_step_boundary(self, key_fn) -> Tuple[Dict[str, int], int]:
        """Dùng để nuôi buff — CHỈ đếm records kể từ watermark step trước."""
        return self._filter_and_count(self._records[self._step_boundary:], key_fn)

    def full_history_counts(self, key_fn) -> Tuple[Dict[str, int], int]:
        """Dùng cho report — đếm TOÀN BỘ records kể từ lần reset() gần nhất."""
        return self._filter_and_count(self._records, key_fn)

    def summary(self) -> Dict[str, Dict[str, dict]]:
        """
        Breakdown chi tiết theo trend -> action_type, CHỈ cho task=action,
        chỉ tính trên records đã pass TOÀN BỘ gate (well_formed +
        semantic_passed + task_passed=True) — đúng quy ước report của
        StatsCollector v1 (freq_within_trend, avg_r_multiple RAW trước
        phí, win_rate, avg_rr, phân phối rr). Dùng field `r_multiple`
        (raw) chứ KHÔNG dùng `outcome` (đã trừ phí, chỉ dùng để tính
        reward) — win_rate/avg_R ở đây là số liệu P&L thô để đọc, tách
        biệt khỏi con số dùng nội bộ để tối ưu.
        """
        by_trend_total: Dict[str, int] = defaultdict(int)
        raw: Dict[str, Dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"count": 0, "r_multiples": [], "rrs": []})
        )
        for r in self._records:
            if r.trend is None:
                continue
            if not (r.well_formed and r.semantic_passed and r.task_passed is True):
                continue
            by_trend_total[r.trend] += 1
            entry = raw[r.trend][r.action_type]
            entry["count"] += 1
            if r.r_multiple is not None:
                entry["r_multiples"].append(r.r_multiple)
            if r.rr is not None:
                entry["rrs"].append(r.rr)

        result: Dict[str, Dict[str, dict]] = {}
        for trend, actions in raw.items():
            result[trend] = {}
            total = by_trend_total[trend]
            for action_type, entry in actions.items():
                rms = entry["r_multiples"]
                rrs = entry["rrs"]
                avg_r = sum(rms) / len(rms) if rms else None
                win_rate = (sum(1 for x in rms if x > 0) / len(rms)) if rms else None
                avg_rr = sum(rrs) / len(rrs) if rrs else None
                result[trend][action_type] = {
                    "count": entry["count"],
                    "freq_within_trend": entry["count"] / total if total else 0.0,
                    "avg_r_multiple": avg_r,
                    "win_rate": win_rate,
                    "avg_rr": avg_rr,
                    "rr_distribution": dict(sorted(Counter(rrs).items())) if rrs else None,
                }
        return result

    def well_form_rate_by_intended_action(self) -> Dict[str, Dict[str, Any]]:
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "well_formed": 0})
        for r in self._records:
            if r.intended_action_type is None:
                continue
            entry = counts[r.intended_action_type]
            entry["total"] += 1
            if r.well_formed:
                entry["well_formed"] += 1
        return {
            a: {**e, "well_form_rate": (e["well_formed"] / e["total"] if e["total"] else 0.0)}
            for a, e in counts.items()
        }

    def print_summary(self) -> None:
        print("=== [reward v2] StatsCollectorV2 summary ===")
        for task_id in TASKS:
            n_task = sum(1 for r in self._records if r.task_id == task_id)
            n_wf = sum(1 for r in self._records if r.task_id == task_id and r.well_formed)
            n_sem = sum(1 for r in self._records if r.task_id == task_id and r.well_formed and r.semantic_passed)
            print(f"\n--- task={task_id} (n={n_task}) ---")
            if n_task:
                print(f"  well_form_rate = {n_wf / n_task * 100:.1f}%")
            if n_wf:
                print(f"  semantic_pass_rate (trong số well-formed) = {n_sem / n_wf * 100:.1f}%")

        print("\n-- Chi tiết theo trend -> action (task=action, đã pass toàn bộ gate) --")
        detail = self.summary()
        if not detail:
            print("  (chưa có mẫu nào pass gate ở task=action)")
        for trend, actions in detail.items():
            print(f"trend={trend}")
            for action_type, stat in actions.items():
                avg_r = f"{stat['avg_r_multiple']:.2f}" if stat["avg_r_multiple"] is not None else "-"
                win_rate = f"{stat['win_rate'] * 100:.0f}%" if stat["win_rate"] is not None else "-"
                avg_rr = f"{stat['avg_rr']:.2f}" if stat.get("avg_rr") is not None else "-"
                line = (
                    f"  {action_type:<12} count={stat['count']:<6} freq={stat['freq_within_trend']*100:5.1f}%  "
                    f"avg_R={avg_r:>6}  win_rate={win_rate:>4}  avg_RR={avg_rr:>5}"
                )
                dist = stat.get("rr_distribution")
                if dist:
                    dist_str = " ".join(f"{k}:{v}" for k, v in dist.items())
                    line += f"  rr_dist=[{dist_str}]"
                print(line)

        print("\n-- Action group counts (7 nhóm, toàn bộ lịch sử từ lần reset gần nhất) --")
        action_counts, action_total = self.full_history_counts(TASK_ACTION, key_fn=lambda r: r.action_type)
        for g in GROUPS_ACTION:
            n = action_counts.get(g, 0)
            ratio = n / action_total if action_total else 0.0
            print(f"  {g:<12} count={n:<6} ratio={ratio * 100:5.1f}%")

        print("\n-- Zone group counts (HAS_ZONE/NO_ZONE) --")
        zone_counts, zone_total = self.full_history_counts(
            TASK_ZONE,
            key_fn=lambda r: "HAS_ZONE" if r.has_zone else ("NO_ZONE" if r.has_zone is False else None),
        )
        for g in GROUPS_ZONE:
            n = zone_counts.get(g, 0)
            ratio = n / zone_total if zone_total else 0.0
            print(f"  {g:<12} count={n:<6} ratio={ratio * 100:5.1f}%")

        print("\n-- Well-form rate theo Ý ĐỊNH action (kể cả parse fail) --")
        for action, stat in sorted(self.well_form_rate_by_intended_action().items()):
            print(
                f"  {action:<12} total={stat['total']:<6} well_formed={stat['well_formed']:<6} "
                f"rate={stat['well_form_rate'] * 100:5.1f}%"
            )

    def to_list(self) -> List[dict]:
        return [asdict(r) for r in self._records]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"records": self.to_list()}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "StatsCollectorV2":
        collector = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("records", []):
                d.setdefault("r_multiple", None)   # tương thích ngược file stats cũ chưa có field này
                collector.log(TaskRolloutMeta(**d))
        return collector

    @classmethod
    def merge_from_files(cls, paths) -> "StatsCollectorV2":
        collector = cls()
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for d in data.get("records", []):
                d.setdefault("r_multiple", None)   # tương thích ngược file stats cũ chưa có field này
                collector.log(TaskRolloutMeta(**d))
        return collector