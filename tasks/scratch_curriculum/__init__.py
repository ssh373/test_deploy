from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .walk_001 import K1WalkControllerCfg


@configclass
class K1ScratchCurriculumWalk001ControllerCfg(K1WalkControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/JY_curriculum_k1_walk_001_2026-07-17_20-58-31.pt"


register_task("k1_scratch_curriculum_walk_001", K1ScratchCurriculumWalk001ControllerCfg())
