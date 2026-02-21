from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .locomotion import T1WalkControllerCfg
from .locomotion_k1 import K1WalkControllerCfg

# Register locomotion tasks


@configclass
class T1WalkControllerCfg1(T1WalkControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_walk.pt"

@configclass
class K1WalkControllerCfg1(K1WalkControllerCfg):
    '''Human-like walk for K1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_walk_002_2026-02-21_00-18-50.pt"

register_task(
    "t1_walk", T1WalkControllerCfg1())
register_task("k1_walk", K1WalkControllerCfg1())
