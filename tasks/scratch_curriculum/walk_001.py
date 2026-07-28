from __future__ import annotations

import math
import os
from dataclasses import MISSING

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    PrepareStateCfg,
    PolicyCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass

POLICY_JOINT_NAMES = [
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
]

K1_22DOF_DEFAULT_JOINT_POS = [
    0.0, 0.0,
    0.2, -1.3, 0.0, -0.5,
    0.2, 1.3, 0.0, 0.5,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
]

K1_22DOF_STIFFNESS = [
    4.0, 4.0,
    4.0, 4.0, 4.0, 4.0,
    4.0, 4.0, 4.0, 4.0,
    80., 80.0, 80., 80., 30., 30.,
    80., 80.0, 80., 80., 30., 30.,
]

K1_22DOF_DAMPING = [
    1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]

K1_22DOF_PREPARE_STIFFNESS = [
    40.0, 40.0,
    40.0, 50.0, 20.0, 20.0,
    40.0, 50.0, 20.0, 20.0,
    350.0, 350.0, 180.0, 350.0, 250.0, 250.0,
    350.0, 350.0, 180.0, 350.0, 250.0, 250.0,
]

K1_22DOF_PREPARE_DAMPING = [
    1.5, 1.5,
    0.5, 1.5, 0.2, 0.2,
    0.5, 1.5, 0.2, 0.2,
    7.5, 7.5, 3.0, 5.5, 5.0, 5.0,
    7.5, 7.5, 3.0, 5.5, 5.0, 5.0,
]


class K1ScratchCurriculumWalkPolicy(Policy):
    def __init__(self, cfg: K1ScratchCurriculumWalkPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.controller = controller
        self.robot = controller.robot
        self.device = torch.device(cfg.device)

        policy_path = cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, policy_path)

        try:
            self._model: torch.jit.ScriptModule = torch.jit.load(policy_path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TorchScript policy: {policy_path}\n"
                "Use exported *_jit.pt, not raw RSL-RL model_*.pt."
            ) from exc

        self._model.to(self.device).eval()
        self.robot.data.to(self.device)

        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)

        self.policy_joint_idx = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in cfg.policy_joint_names],
            dtype=torch.long,
            device=self.device,
        )

        self.default_internal_parameters = torch.tensor(
            cfg.default_internal_parameters,
            dtype=torch.float32,
            device=self.device,
        )

        self.last_action = torch.zeros(len(cfg.policy_joint_names), dtype=torch.float32, device=self.device)
        self.gait_process = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.last_action.zero_()
        self.gait_process.zero_()

    def _velocity_command(self) -> torch.Tensor:
        if self.controller.vel_command is None:
            vx, vy, wz = self.cfg.default_velocity_command
        else:
            cmd = self.controller.vel_command
            vx = max(-cmd.vx_max, min(cmd.vx_max, cmd.lin_vel_x))
            vy = max(-cmd.vy_max, min(cmd.vy_max, cmd.lin_vel_y))
            wz = max(-cmd.vyaw_max, min(cmd.vyaw_max, cmd.ang_vel_yaw))

        return torch.tensor([vx, vy, wz], dtype=torch.float32, device=self.device)

    def _command(self) -> torch.Tensor:
        velocity_command = self._velocity_command()
        internal_parameters = self.default_internal_parameters.clone()

        if torch.linalg.norm(velocity_command, ord=2) < self.cfg.stand_command_threshold:
            internal_parameters[0] = 0.0

        return torch.cat([velocity_command, internal_parameters], dim=0)

    def _gait_phase(self, gait_frequency: torch.Tensor) -> torch.Tensor:
        self.gait_process = torch.fmod(
            self.gait_process + self.controller.cfg.policy_dt * gait_frequency,
            1.0,
        )
        return torch.stack(
            (
                torch.cos(2.0 * math.pi * self.gait_process),
                torch.sin(2.0 * math.pi * self.gait_process),
            ),
            dim=0,
        )

    def compute_observation(self) -> torch.Tensor:
        command = self._command()
        gait_phase = self._gait_phase(command[3])

        base_quat = self.robot.data.root_quat_w
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print("\nFalling detected, stopping policy.")
            self.controller.stop()

        joint_pos = self.robot.data.joint_pos[self.policy_joint_idx]
        joint_vel = self.robot.data.joint_vel[self.policy_joint_idx]
        default_joint_pos = self.default_joint_pos[self.policy_joint_idx]

        obs = torch.cat(
            [
                command,
                gait_phase,
                projected_gravity,
                self.robot.data.root_ang_vel_b,
                joint_pos - default_joint_pos,
                joint_vel,
                self.last_action,
            ],
            dim=0,
        )

        if obs.numel() != self.cfg.obs_dim:
            raise RuntimeError(f"Expected obs dim {self.cfg.obs_dim}, got {obs.numel()}")

        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        with torch.no_grad():
            obs = self.compute_observation()
            action = self._model(obs).reshape(-1)

        if action.numel() != len(self.cfg.policy_joint_names):
            raise RuntimeError(f"Expected action dim {len(self.cfg.policy_joint_names)}, got {action.numel()}")

        action = torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)
        self.last_action = action.detach().clone()

        dof_targets = self.default_joint_pos.clone()
        dof_targets[self.policy_joint_idx] = self.default_joint_pos[self.policy_joint_idx] + action
        return dof_targets


@configclass
class K1ScratchCurriculumWalkPolicyCfg(PolicyCfg):
    constructor = K1ScratchCurriculumWalkPolicy
    checkpoint_path: str = MISSING  # type: ignore

    obs_dim: int = 54
    clip_actions: float = 1.0
    enable_safety_fallback: bool = False

    policy_joint_names: list[str] = POLICY_JOINT_NAMES
    default_velocity_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stand_command_threshold: float = 0.05
    default_internal_parameters: tuple[float, float, float, float, float, float, float] = (
        1.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


@configclass
class K1WalkControllerCfg(ControllerCfg):
    policy_dt: float = 0.02

    robot = K1_CFG.replace(
        default_joint_pos=K1_22DOF_DEFAULT_JOINT_POS,
        joint_stiffness=K1_22DOF_STIFFNESS,
        joint_damping=K1_22DOF_DAMPING,
        prepare_state=PrepareStateCfg(
            stiffness=K1_22DOF_PREPARE_STIFFNESS,
            damping=K1_22DOF_PREPARE_DAMPING,
            joint_pos=K1_22DOF_DEFAULT_JOINT_POS,
        ),
    )

    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=1.5,
        vy_max=1.5,
        vyaw_max=1.6,
    )

    policy: K1ScratchCurriculumWalkPolicyCfg = K1ScratchCurriculumWalkPolicyCfg(
        policy_joint_names=POLICY_JOINT_NAMES,
    )

    mujoco: MujocoControllerCfg = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.58],
        init_quat=[1.0, 0.0, 0.0, 0.0],
        decimation=10,
    )
