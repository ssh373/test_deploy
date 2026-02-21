from __future__ import annotations
from dataclasses import MISSING
import os
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import K1_CFG, T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math


class LocomotionPolicy(Policy):
    """walking policy with observation history."""

    def __init__(self, cfg: LocomotionPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        # Load policy model
        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            # Try relative to task directory
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)

        self._model: torch.jit.ScriptModule = torch.jit.load(
            policy_path, map_location="cpu")
        self._model.eval()

        # Observation and action parameters
        self.actor_obs_history_length = cfg.actor_obs_history_length
        self.action_scale = (
            cfg.action_scale * self.robot.effort_limit / self.robot.joint_stiffness
        ).to(self.cfg.device)

        # Initialize buffers
        self.obs_history = None
        self.last_action = torch.zeros(
            len(self.cfg.policy_joint_names), dtype=torch.float32)

        self.fixed_upper_body_pos = torch.tensor([
            0.0, 0.0,               # Head (Yaw, Pitch)
            0.2, -1.3, 0.0, -0.5,     # Left Arm (Shoulder P, R, Elbow P, Y)
            0.2, 1.3, 0.0, 0.5,      # Right Arm (Shoulder P, R, Elbow P, Y)
        ], dtype=torch.float32)

        self.real2sim_joint_map = torch.tensor([
            self.robot.cfg.joint_names.index(name)
            for name in self.cfg.policy_joint_names
        ], dtype=torch.long)

        self.upper_joint_names = [
            "AAHead_yaw", "Head_pitch",
            "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
            "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        ]
        self.upper_idx = torch.tensor(
            [self.robot.cfg.joint_names.index(n) for n in self.upper_joint_names],
            dtype=torch.long
        )
        self._step = 0


    def reset(self) -> None:
        """Initialize policy state."""
        pass

    def compute_observation(self) -> torch.Tensor:
        """Compute current observation following sim2sim.py pattern."""
        # Get robot state
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_quat = self.robot.data.root_quat_w

        base_lin_vel = self.robot.data.root_lin_vel_b

        base_ang_vel = self.robot.data.root_ang_vel_b

        # Project gravity vector into base frame
        # gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=base_quat.device)

        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)
        self._last_projected_gravity = projected_gravity.detach()

        if self.cfg.enable_safety_fallback:
            # fall detection: stop if falling
            if projected_gravity[2] > -0.5:
                print("\nFalling detected, stopping policy for safety. "
                      "You can disable safety fallback by setting "
                      f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                      "to False.")
                self.controller.stop()

        # Get velocity commands
        lin_vel_x = self.controller.vel_command.lin_vel_x
        lin_vel_y = self.controller.vel_command.lin_vel_y
        ang_vel_yaw = self.controller.vel_command.ang_vel_yaw

        default_joint_pos_sim = self.robot.default_joint_pos
        mapped_default_pos = default_joint_pos_sim[self.real2sim_joint_map]
        mapped_dof_pos = dof_pos[self.real2sim_joint_map]
        mapped_dof_vel = dof_vel[self.real2sim_joint_map]

        # Build observation: [
        #   ang_vel(3),
        #   projected_gravity(3),
        #   commands(3),
        #   joint_pos(num_action),
        #   joint_vel(num_action),
        #   actions(num_action)]
        self._last_cmd = (lin_vel_x, lin_vel_y, ang_vel_yaw)

        obs = torch.cat([
            base_ang_vel,
            projected_gravity,
            torch.tensor(
                [lin_vel_x, lin_vel_y, ang_vel_yaw], dtype=torch.float32),
            (mapped_dof_pos - mapped_default_pos) * 1.0,
            mapped_dof_vel * self.cfg.obs_dof_vel_scale,
            self.last_action * 1.0
        ], dim=0)

        return obs

    def inference(self) -> torch.Tensor:
        """Compute action from policy."""
        # Compute current observation
        obs = self.compute_observation()

        if self.obs_history is None:
            self.obs_history = torch.zeros(
                self.actor_obs_history_length,
                obs.numel(),
                dtype=torch.float32
            )

        # Update observation history (roll and append)
        self.obs_history = self.obs_history.roll(shifts=-1, dims=0)
        self.obs_history[-1] = obs

        # Get action from policy
        with torch.no_grad():
            action = self._model(self.obs_history.flatten()).squeeze(0)
        self._step += 1
        
        # Store action for next step
        self.last_action = action.clone()

        # default_joint_pos = self.robot.default_joint_pos

        # dof_targets = default_joint_pos.clone()
        # dof_targets.scatter_reduce_(
        #     0,
        #     self.real2sim_joint_map,
        #     action * self.action_scale,
        #     reduce='sum')
        
        # 1. 전체 타겟을 로봇의 default_joint_pos로 초기화 (상체 고정값 포함됨)
        dof_targets = self.robot.default_joint_pos.clone()

        # 2. 하체 관절에 맞는 스케일만
        # action은 12개이고, action_scale은 22개이므로 인덱스 매핑이 필요합니다.
        # 3. 12개 관절에 대한 액션 계산
        mapped_action_scale = self.action_scale[self.real2sim_joint_map]
        mapped_action = action * mapped_action_scale
        
        # 4. 계산된 값을 전체 dof_targets의 해당 위치에 삽입
        dof_targets.scatter_(0, self.real2sim_joint_map, 
                             self.robot.default_joint_pos[self.real2sim_joint_map] + mapped_action)
        
        # 5. 상체 고정
        dof_targets[self.upper_idx] = self.fixed_upper_body_pos
        return dof_targets


@configclass
class LocomotionPolicyCfg(PolicyCfg):
    constructor = LocomotionPolicy
    checkpoint_path: str = MISSING  # type: ignore
    # actor_obs_history_length: int = 10
    actor_obs_history_length: int = 1
    action_scale: float = 0.25
    obs_dof_vel_scale: float = 1.0
    policy_joint_names: list[str] = MISSING  # type: ignore
    enable_safety_fallback: bool = False


@configclass
class K1WalkControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        default_joint_pos=[
            0, 0,
            0.2, -1.3, 0,  -0.5,
            0.2,  1.3, 0,  0.5,
            -0.2, 0, 0, 0.4, -0.2, 0.,
            -0.2, 0, 0, 0.4, -0.2, 0.
        ],
        joint_stiffness=[
            4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            80., 80.0, 80., 80., 30., 30.,
            80., 80.0, 80., 80., 30., 30.,
        ],
        joint_damping=[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, # 2.0에서 5.0으로 상향
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ]
    )
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=1.0,
        vy_max=1.0,
        vyaw_max=1.0,
    )
    policy: LocomotionPolicyCfg = LocomotionPolicyCfg(
        obs_dof_vel_scale=1.0,
        policy_joint_names=[
            # "AAHead_yaw",
            # "Head_pitch",
            # "ALeft_Shoulder_Pitch",
            # "Left_Shoulder_Roll",
            # "Left_Elbow_Pitch",
            # "Left_Elbow_Yaw",
            # "ARight_Shoulder_Pitch",
            # "Right_Shoulder_Roll",
            # "Right_Elbow_Pitch",
            # "Right_Elbow_Yaw",
            
            # "Left_Hip_Pitch",
            # "Left_Hip_Roll",
            # "Left_Hip_Yaw",
            # "Left_Knee_Pitch",
            # "Left_Ankle_Pitch",
            # "Left_Ankle_Roll",
            # "Right_Hip_Pitch",
            # "Right_Hip_Roll",
            # "Right_Hip_Yaw",
            # "Right_Knee_Pitch",
            # "Right_Ankle_Pitch",
            # "Right_Ankle_Roll",

            "Left_Hip_Pitch", "Right_Hip_Pitch",
            "Left_Hip_Roll",  "Right_Hip_Roll",
            "Left_Hip_Yaw",   "Right_Hip_Yaw",
            "Left_Knee_Pitch","Right_Knee_Pitch",
            "Left_Ankle_Pitch","Right_Ankle_Pitch",
            "Left_Ankle_Roll","Right_Ankle_Roll",
        ],
    )