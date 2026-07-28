from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .goto_k1 import K1GoToControllerCfg


@configclass
class K1GoToDeployCfg(K1GoToControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_goto_jit.pt"


register_task("k1_goto", K1GoToDeployCfg())
