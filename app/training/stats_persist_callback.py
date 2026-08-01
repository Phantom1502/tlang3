from transformers import TrainerCallback
from app.config.schema import RoundConfig
from app.training.reward.zone_buff_controller import EMABuffController
class StatsPersistCallbackV2(TrainerCallback):
    def __init__(self, round_config: RoundConfig):
        self.round_config = round_config
        self.buff_controller = EMABuffController(groups=list(round_config.zone_buffs.keys()), namespace="zone_buff")
        self.buff_controller.init(round_config)
        
    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, **kwargs):
        pass
    
    def on_save(self, args, state, control, **kwargs):
        pass
    
    def on_train_end(self, args, state, control, **kwargs):
        pass