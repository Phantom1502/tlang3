import os
import logging
import os

logger = logging.getLogger("app.train.reward.stats_persist_callback")

from transformers import TrainerCallback
from app.config.schema import RoundConfig
from app.training.reward.zone_buff_controller import EMABuffController
from app.training.reward.stats_collector import StatsCollector

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def _stats_path_for_rank(output_dir: str, round_id: str, rank: int) -> str:
    return os.path.join(output_dir, f"{round_id}_stats_rank{rank}.json")

class StatsPersistCallbackV2(TrainerCallback):
    def __init__(self, buff_controller: EMABuffController, stats_collector: StatsCollector, round_config: RoundConfig):
        self.buff_controller = buff_controller
        self.stats_collector = stats_collector
        self.round_config = round_config
        
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.stats_path = _stats_path_for_rank(self.round_config.output_dir, self.round_config.round_id, rank)

    def on_step_end(self, args, state, control, **kwargs):
        zone_counts, total = self.stats_collector.counts_since_step_boundary(
            task_id="zone", key_fn=lambda r: r.zone_type
        )
        self.buff_controller.on_step_end(
            round_config=self.round_config,
            counts=zone_counts,
            total=total
        )
        self.stats_collector.mark_step_boundary()

    def on_log(self, args, state, control, **kwargs):
        for group, metrics in self.buff_controller.snapshot().items():
            print(f"{group}: ema_ratio={metrics['ema_ratio']:.4f}, buff={metrics['buff']:.4f}, prev_error={metrics['prev_error']:.4f}")
    
    def on_save(self, args, state, control, **kwargs):
        n_records = len(self.stats_collector._records)
        print(f"\n=== [step={state.global_step}] Chu kỳ report vừa xong ({n_records} record) ===")
        self.stats_collector.print_summary()
        print(f"action_buff_controller hiện tại: {self.buff_controller.snapshot()}")
        print(f"zone_buff_controller hiện tại: {self.buff_controller.snapshot()}")
        print(f"rr_entropy_controller_v2 hiện tại: {self.buff_controller.snapshot()}\n")

        self.stats_collector.save(self.stats_path)
        self.stats_collector.reset()

        ckpt_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            self.buff_controller.save(os.path.join(ckpt_dir, "action_buff_state_v2.json"))
            self.buff_controller.save(os.path.join(ckpt_dir, "zone_buff_state_v2.json"))
            self.buff_controller.save(os.path.join(ckpt_dir, "rr_entropy_state_v2.json"))
            logger.info(f"Đã lưu action/zone_buff_state + rr_entropy_state -> {ckpt_dir}/")
        else:
            logger.warning(f"Checkpoint dir {ckpt_dir} chưa tồn tại lúc on_save — bỏ qua lưu state.")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\n=== [train_end] Chu kỳ report cuối cùng ===")
        self.stats_collector.print_summary()
        print(f"action_buff_controller cuối: {self.buff_controller.snapshot()}")
        print(f"zone_buff_controller cuối: {self.buff_controller.snapshot()}")
        print(f"rr_entropy_controller_v2 cuối: {self.buff_controller.snapshot()}\n")
        self.stats_collector.save(self.stats_path)