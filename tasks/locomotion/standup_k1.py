from __future__ import annotations

import os
from dataclasses import MISSING

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    PolicyCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass


class StandupPolicy(Policy):
    """K1 stand-up policy ported from IsaacGym/B-Human RL get-up observation layout."""

    def __init__(self, cfg: StandupPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)

        self._model: torch.jit.ScriptModule = torch.jit.load(policy_path, map_location="cpu")
        self._model.eval()

        self.action_scale = cfg.action_scale
        self.obs_dof_vel_scale = cfg.obs_dof_vel_scale
        self.clip_actions = cfg.clip_actions

        self.real2sim_joint_map = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in self.cfg.policy_joint_names],
            dtype=torch.long,
        )
        self.offset_joint_pos = torch.tensor(cfg.default_joint_pos, dtype=torch.float32)
        self.last_request = torch.zeros(len(self.cfg.policy_joint_names), dtype=torch.float32)

    def reset(self) -> None:
        self.last_request.zero_()

    def _compute_phase(self) -> torch.Tensor:
        elapsed_s = float(getattr(self.controller, "_elapsed_s", 0.0))
        denom = max(self.cfg.max_stand_up_time_s, 1e-4)
        phase = min(max(elapsed_s / denom, 0.0), 1.0)
        return torch.tensor([phase], dtype=torch.float32)

    def compute_observation(self) -> torch.Tensor:
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_b

        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=base_quat.device)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print(
                "\nFalling detected, stopping policy for safety. "
                "You can disable safety fallback by setting "
                f"{self.cfg.__class__.__name__}.enable_safety_fallback to False."
            )
            self.controller.stop()

        mapped_dof_pos = dof_pos[self.real2sim_joint_map]
        mapped_dof_vel = dof_vel[self.real2sim_joint_map]

        obs = torch.cat(
            [
                projected_gravity,
                base_ang_vel,
                self._compute_phase(),
                mapped_dof_pos - self.offset_joint_pos,
                mapped_dof_vel * self.obs_dof_vel_scale,
                self.last_request,
            ],
            dim=0,
        )
        return obs.clamp(-100.0, 100.0)

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()
        with torch.no_grad():
            action = self._model(obs).squeeze(0)
            action = torch.clamp(action, -self.clip_actions, self.clip_actions)

        dof_targets = self.robot.default_joint_pos.clone()
        mapped_action = action * self.action_scale
        dof_targets[self.real2sim_joint_map] = self.offset_joint_pos + mapped_action

        # Next observation uses previous requested joint target offset.
        self.last_request = dof_targets[self.real2sim_joint_map] - self.offset_joint_pos
        return dof_targets


@configclass
class StandupPolicyCfg(PolicyCfg):
    constructor = StandupPolicy
    checkpoint_path: str = MISSING  # type: ignore
    action_scale: float = 1.0
    obs_dof_vel_scale: float = 0.1
    clip_actions: float = 3.0
    max_stand_up_time_s: float = 6.0
    enable_safety_fallback: bool = False
    default_joint_pos: list[float] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.25,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.25,
        0.0,
    ]
    policy_joint_names: list[str] = MISSING  # type: ignore


@configclass
class K1StandupControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        default_joint_pos=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,
        ],
        joint_stiffness=[
            10.0,
            10.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            80.0,
            80.0,
            80.0,
            80.0,
            35.0,
            35.0,
            80.0,
            80.0,
            80.0,
            80.0,
            35.0,
            35.0,
        ],
        joint_damping=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            4.0,
            4.0,
            4.0,
            4.0,
            1.0,
            1.0,
            4.0,
            4.0,
            4.0,
            4.0,
            1.0,
            1.0,
        ],
    )
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=0.0,
        vy_max=0.0,
        vyaw_max=0.0,
    )
    policy: StandupPolicyCfg = StandupPolicyCfg(
        policy_joint_names=[
            "AAHead_yaw",
            "Head_pitch",
            "ALeft_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "ARight_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ],
    )
