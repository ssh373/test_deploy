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


class KickPolicy(Policy):
    """Deployment-oriented K1 walking policy with fixed upper body."""

    def __init__(self, cfg: KickPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)

        self._model: torch.jit.ScriptModule = torch.jit.load(policy_path, map_location="cpu")
        self._model.eval()

        self.actor_obs_history_length = cfg.actor_obs_history_length
        # Match training-time action application:
        # dof_target = default_dof_pos + action_scale * action
        self.action_scale = cfg.action_scale

        self.obs_history = None
        self.last_action = torch.zeros(len(self.cfg.policy_joint_names), dtype=torch.float32)

        self.real2sim_joint_map = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in self.cfg.policy_joint_names],
            dtype=torch.long,
        )

        # Keep only arms fixed. Head joints are intentionally excluded so
        # external trackers (e.g. track-ball) can command neck motion.
        self.upper_joint_names = [
            "ALeft_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "ARight_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
        ]
        self.upper_idx = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in self.upper_joint_names],
            dtype=torch.long,
        )
        self.fixed_upper_body_pos = torch.tensor(
            [
                0.0,
                -1.35,
                0.0,
                0.0,
                0.0,
                1.35,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )
        self._ball_body_id: int | None = None
        self._ball_warned = False

    def reset(self) -> None:
        if not self.cfg.reset_ball_on_start:
            return
        if not (hasattr(self.controller, "mj_model") and hasattr(self.controller, "mj_data")):
            return
        try:
            import mujoco

            joint_id = mujoco.mj_name2id(
                self.controller.mj_model,  # type: ignore[attr-defined]
                mujoco.mjtObj.mjOBJ_JOINT,
                self.cfg.ball_joint_name,
            )
            if joint_id < 0:
                return

            qpos_adr = int(self.controller.mj_model.jnt_qposadr[joint_id])  # type: ignore[attr-defined]
            qvel_adr = int(self.controller.mj_model.jnt_dofadr[joint_id])  # type: ignore[attr-defined]

            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w
            rel_b = torch.tensor(
                [self.cfg.ball_spawn_rel_xy[0], self.cfg.ball_spawn_rel_xy[1], 0.0],
                dtype=torch.float32,
                device=root_quat_w.device,
            )
            rel_w = lab_math.quat_apply(root_quat_w, rel_b)
            ball_pos_w = (root_pos_w + rel_w).detach().cpu().numpy().astype("float64")
            ball_pos_w[2] = float(self.cfg.ball_spawn_height)

            self.controller.mj_data.qpos[qpos_adr:qpos_adr + 3] = ball_pos_w  # type: ignore[attr-defined]
            self.controller.mj_data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]  # type: ignore[attr-defined]
            self.controller.mj_data.qvel[qvel_adr:qvel_adr + 6] = 0.0  # type: ignore[attr-defined]
            mujoco.mj_forward(self.controller.mj_model, self.controller.mj_data)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"Warning: failed to reset ball pose on start: {e}")

    def _get_ball_relative_xy(self, base_quat: torch.Tensor) -> torch.Tensor:
        """Return ball relative xy in base frame.

        Priority:
        1) externally injected controller.ball_rel_xy (e.g. vision node),
        2) MuJoCo body position lookup by name,
        3) zero fallback.
        """
        if hasattr(self.controller, "ball_rel_xy"):
            rel_xy = torch.as_tensor(
                getattr(self.controller, "ball_rel_xy"),
                dtype=torch.float32,
                device=base_quat.device,
            )
            return rel_xy[:2]

        if hasattr(self.controller, "mj_model") and hasattr(self.controller, "mj_data"):
            try:
                if self._ball_body_id is None:
                    import mujoco

                    self._ball_body_id = mujoco.mj_name2id(
                        self.controller.mj_model,  # type: ignore[attr-defined]
                        mujoco.mjtObj.mjOBJ_BODY,
                        self.cfg.ball_body_name,
                    )
                if self._ball_body_id >= 0:
                    ball_pos_w_np = self.controller.mj_data.xpos[self._ball_body_id]  # type: ignore[attr-defined]
                    ball_pos_w = torch.as_tensor(
                        ball_pos_w_np, dtype=torch.float32, device=base_quat.device
                    )
                    root_pos_w = self.robot.data.root_pos_w.to(base_quat.device)
                    ball_rel_w = ball_pos_w - root_pos_w
                    ball_rel_b = lab_math.quat_apply_inverse(base_quat, ball_rel_w)
                    return ball_rel_b[:2]
            except Exception:
                pass

        fallback_rel_xy = torch.as_tensor(
            self.cfg.missing_ball_rel_xy,
            dtype=torch.float32,
            device=base_quat.device,
        )[:2]

        if not self._ball_warned:
            print(
                f"Warning: ball relative xy unavailable. "
                f"Using fallback {fallback_rel_xy.tolist()}. "
                f"Set controller.ball_rel_xy or add body '{self.cfg.ball_body_name}' in MuJoCo model."
            )
            self._ball_warned = True
        return fallback_rel_xy

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

        ball_rel_xy = self._get_ball_relative_xy(base_quat)

        mapped_default_pos = self.robot.default_joint_pos[self.real2sim_joint_map]
        mapped_dof_pos = dof_pos[self.real2sim_joint_map]
        mapped_dof_vel = dof_vel[self.real2sim_joint_map]

        obs = torch.cat(
            [
                projected_gravity,
                base_ang_vel,
                ball_rel_xy,
                mapped_dof_pos - mapped_default_pos,
                mapped_dof_vel * self.cfg.obs_dof_vel_scale,
                self.last_action,
            ],
            dim=0,
        )

        return obs.clamp(-100.0, 100.0)

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()

        if self.obs_history is None:
            self.obs_history = torch.zeros(
                self.actor_obs_history_length,
                obs.numel(),
                dtype=torch.float32,
            )

        self.obs_history = self.obs_history.roll(shifts=-1, dims=0)
        self.obs_history[-1] = obs

        with torch.no_grad():
            action = self._model(self.obs_history.flatten()).squeeze(0)
            action = torch.clamp(action, -100.0, 100.0)

        self.last_action = action.clone()

        dof_targets = self.robot.default_joint_pos.clone()
        mapped_action = action * self.action_scale
        dof_targets[self.real2sim_joint_map] = (
            self.robot.default_joint_pos[self.real2sim_joint_map] + mapped_action
        )
        if self.cfg.freeze_lower_body:
            # Temporary test mode: keep policy-controlled lower-body joints fixed.
            dof_targets[self.real2sim_joint_map] = self.robot.default_joint_pos[
                self.real2sim_joint_map
            ]
        dof_targets[self.upper_idx] = self.fixed_upper_body_pos
        return dof_targets


@configclass
class KickPolicyCfg(PolicyCfg):
    constructor = KickPolicy
    checkpoint_path: str = MISSING  # type: ignore
    actor_obs_history_length: int = 1
    action_scale: float = 1.0
    obs_dof_vel_scale: float = 0.1
    ball_body_name: str = "ball"
    ball_joint_name: str = "ball_freejoint"
    missing_ball_rel_xy: list[float] = [0.5, 0.0]
    reset_ball_on_start: bool = True
    ball_spawn_rel_xy: list[float] = [0.5, 0.0]
    ball_spawn_height: float = 0.075
    freeze_lower_body: bool = True
    policy_joint_names: list[str] = MISSING  # type: ignore
    enable_safety_fallback: bool = False


@configclass
class K1KickControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        mjcf_path="/home/user/booster_deploy/tasks/locomotion/assets/K1_22dof_kick_ball.xml",
        default_joint_pos=[
            0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            -0.2, 0, 0, 0.4, -0.25, 0,
            -0.2, 0, 0, 0.4, -0.25, 0,
        ],
        joint_stiffness=[
            10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 
            10.0, 10.0, 10.0, 10.0,
            80.0, 80.0, 80.0, 80.0, 35.0, 35.0,
            80.0, 80.0, 80.0, 80.0, 35.0, 35.0,
        ],
        joint_damping=[
            1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            4.0, 4.0, 4.0, 4.0, 1.0, 1.0,
            4.0, 4.0, 4.0, 4.0, 1.0, 1.0,
        ],
    )

    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=1.0,
        vy_max=1.0,
        vyaw_max=1.0,
    )

    policy: KickPolicyCfg = KickPolicyCfg(
        obs_dof_vel_scale=0.1,
        policy_joint_names=[
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
