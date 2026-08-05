"""MuJoCo deployment registration for the Isaac Lab K1 kick policy."""

from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .kick_k1 import K1IsaacLabKickControllerCfg


@configclass
class K1IsaacLabKickDeployCfg(K1IsaacLabKickControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_kick_jit.pt"


register_task("k1_kick_isaaclab", K1IsaacLabKickDeployCfg())


@configclass
class K1IsaacLabKickHoldDeployCfg(K1IsaacLabKickDeployCfg):
    """MuJoCo diagnostic: use identical physics but disable policy actions."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.hold_default_pose = True


register_task("k1_kick_isaaclab_hold", K1IsaacLabKickHoldDeployCfg())


@configclass
class K1IsaacLabKickClippedDeployCfg(K1IsaacLabKickDeployCfg):
    """Diagnostic: bound an existing unclipped policy at deployment time."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.clip_actions = 1.0


register_task("k1_kick_isaaclab_clipped", K1IsaacLabKickClippedDeployCfg())
