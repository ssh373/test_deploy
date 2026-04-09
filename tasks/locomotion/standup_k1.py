from __future__ import annotations

import math
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


def _deg(v: float) -> float:
    return math.radians(v)


class StandupPolicy(Policy):
    """K1 get-up policy with RLGetUpEngine-like phase state machine."""

    def __init__(self, cfg: StandupPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)
        self._model: torch.jit.ScriptModule = torch.jit.load(policy_path, map_location="cpu")
        self._model.eval()

        self.real2sim_joint_map = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in self.cfg.policy_joint_names],
            dtype=torch.long,
        )
        self.lower_body_idx = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in self.cfg.lower_body_joint_names],
            dtype=torch.long,
        )

        self.offset_joint_pos = torch.tensor(self.cfg.default_joint_pos, dtype=torch.float32)
        self.done_joint_pos = torch.tensor(self.cfg.done_joint_pos, dtype=torch.float32)
        self.fall_angles = torch.tensor(self.cfg.fall_angles, dtype=torch.float32)
        self.recover_front = [
            (torch.tensor(pose, dtype=torch.float32), duration_s)
            for pose, duration_s in self.cfg.recover_front
        ]
        self.recover_back = [
            (torch.tensor(pose, dtype=torch.float32), duration_s)
            for pose, duration_s in self.cfg.recover_back
        ]
        self.recover_normal = [
            (torch.tensor(pose, dtype=torch.float32), duration_s)
            for pose, duration_s in self.cfg.recover_normal
        ]

        self.front_info = self.cfg.front_info
        self.back_info = self.cfg.back_info
        self.front_first_exec_s = self.front_info[0][2]
        self.back_first_exec_s = self.back_info[0][2]

        self.last_request = torch.zeros(len(self.cfg.policy_joint_names), dtype=torch.float32)
        self.last_output = self.robot.default_joint_pos.clone()
        self.recovery_start_joint_pos = self.robot.data.joint_pos.clone()
        self.recover_motion = self.recover_normal
        self.recover_motion_idx = 0
        self.recover_motion_elapsed_s = 0.0
        self.state = "recovery"
        self.state_start_s = 0.0
        self.try_counter = 0
        self.is_front = True
        self.max_stand_up_time_s = self.cfg.front_total_time_s

    def _now(self) -> float:
        return float(getattr(self.controller, "_elapsed_s", 0.0))

    def _stand_executed_s(self) -> float:
        return max(0.0, (self._now() - self.state_start_s) * self.cfg.speed_factor)

    def reset(self) -> None:
        self.last_request.zero_()
        self.last_output = self.robot.data.joint_pos.clone()
        self.try_counter = 0
        self._enter_recovery()

    def _get_torso_xy(self) -> tuple[float, float]:
        base_quat = self.robot.data.root_quat_w.unsqueeze(0)
        roll, pitch, _ = lab_math.euler_xyz_from_quat(base_quat)
        return float(roll[0].item()), float(pitch[0].item())

    def _choose_recover_motion(self) -> list[tuple[torch.Tensor, float]]:
        torso_x, torso_y = self._get_torso_xy()
        if abs(torso_x) >= self.cfg.recover_back_roll_threshold_rad and torso_y < self.cfg.recover_back_pitch_threshold_rad:
            return self.recover_back
        if abs(torso_x) >= self.cfg.recover_front_roll_threshold_rad:
            return self.recover_front
        return self.recover_normal

    def _enter_recovery(self) -> None:
        self.state = "recovery"
        self.state_start_s = self._now()
        self.recover_motion = self._choose_recover_motion()
        self.recover_motion_idx = 0
        self.recover_motion_elapsed_s = 0.0
        self.recovery_start_joint_pos = self.robot.data.joint_pos.clone()

    def _enter_stand_up(self) -> None:
        self.state = "stand_up"
        self.state_start_s = self._now()
        torso_x, torso_y = self._get_torso_xy()
        _ = torso_x
        self.is_front = torso_y > 0.0
        self.max_stand_up_time_s = self.cfg.front_total_time_s if self.is_front else self.cfg.back_total_time_s

    def _enter_break_up(self) -> None:
        self.state = "break_up"
        self.state_start_s = self._now()
        self.try_counter += 1

    def _transition(self) -> None:
        if self.state == "recovery":
            if self.recover_motion_idx >= len(self.recover_motion):
                self._enter_stand_up()
            return

        if self.state == "stand_up":
            stand_elapsed = self._stand_executed_s()
            first_exec = self.front_first_exec_s if self.is_front else self.back_first_exec_s
            if self.cfg.use_break_up and stand_elapsed > first_exec and self._should_break_up(stand_elapsed):
                self._enter_break_up()
                return

            ratio = stand_elapsed / max(self.max_stand_up_time_s, 1.0e-6)
            base_height = float(self.robot.data.root_pos_w[2].item())
            gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.robot.data.root_quat_w.device)
            projected_gravity = lab_math.quat_apply_inverse(self.robot.data.root_quat_w, gravity_w)
            upright = float(projected_gravity[2].item()) < -0.75
            early_done = ratio >= self.cfg.earliest_done_ratio and base_height > self.cfg.min_stand_height_when_done and upright
            if ratio >= 1.0 or early_done:
                self.state = "done"
                self.state_start_s = self._now()
            return

        if self.state == "break_up":
            if (self._now() - self.state_start_s) >= self.cfg.break_up_wait_s:
                if self.try_counter < self.cfg.max_try_counter:
                    self._enter_recovery()
                else:
                    self.state = "done"
                    self.state_start_s = self._now()

    def _recover_target(self) -> torch.Tensor:
        if self.recover_motion_idx >= len(self.recover_motion):
            return self.robot.default_joint_pos.clone()

        target_pose, duration_s = self.recover_motion[self.recover_motion_idx]
        dt = self.cfg.policy_dt
        self.recover_motion_elapsed_s += dt
        ratio = min(max(self.recover_motion_elapsed_s / max(duration_s, 1.0e-6), 0.0), 1.0)

        target = self.recovery_start_joint_pos * (1.0 - ratio) + target_pose * ratio
        if ratio >= 1.0:
            self.recover_motion_idx += 1
            self.recover_motion_elapsed_s = 0.0
            self.recovery_start_joint_pos = target.clone()
        return target

    def _compute_phase(self) -> torch.Tensor:
        if self.state != "stand_up":
            return torch.tensor([0.0], dtype=torch.float32)
        elapsed_s = self._stand_executed_s()
        denom = max(self.max_stand_up_time_s, 1.0e-6)
        return torch.tensor([min(max(elapsed_s / denom, 0.0), 1.0)], dtype=torch.float32)

    def _should_break_up(self, executed_s: float) -> bool:
        info = self.front_info if self.is_front else self.back_info
        torso_x, torso_y = self._get_torso_xy()

        start_idx = 0
        next_idx = len(info) - 1
        for i, item in enumerate(info):
            if item[2] < executed_s:
                start_idx = i
            else:
                next_idx = i
                break

        sx_min, sx_max, sy_min, sy_max, st = info[start_idx][0], info[start_idx][1], info[start_idx][3], info[start_idx][4], info[start_idx][2]
        nx_min, nx_max, ny_min, ny_max, nt = info[next_idx][0], info[next_idx][1], info[next_idx][3], info[next_idx][4], info[next_idx][2]
        if next_idx == start_idx or nt <= st:
            tx_min, tx_max, ty_min, ty_max = sx_min, sx_max, sy_min, sy_max
        else:
            alpha = min(max((executed_s - st) / (nt - st), 0.0), 1.0)
            tx_min = sx_min * (1.0 - alpha) + nx_min * alpha
            tx_max = sx_max * (1.0 - alpha) + nx_max * alpha
            ty_min = sy_min * (1.0 - alpha) + ny_min * alpha
            ty_max = sy_max * (1.0 - alpha) + ny_max * alpha

        return not (tx_min <= torso_x <= tx_max and ty_min <= torso_y <= ty_max)

    def _compute_rl_observation(self) -> torch.Tensor:
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
                mapped_dof_vel * self.cfg.obs_dof_vel_scale,
                self.last_request,
            ],
            dim=0,
        )
        return obs.clamp(-100.0, 100.0)

    def _break_up_target(self) -> torch.Tensor:
        dt = self.cfg.policy_dt
        speed_limit = self.cfg.break_up_joint_speed_rad_s * dt
        dof_pos = self.robot.data.joint_pos

        target = self.fall_angles.clone()
        torso_x, torso_y = self._get_torso_xy()
        _ = torso_x
        target[1] = -self.cfg.break_up_head_angle_rad if torso_y > 0.0 else self.cfg.break_up_head_angle_rad

        req = self.last_output.clone()
        req[self.lower_body_idx] = dof_pos[self.lower_body_idx]
        lower = dof_pos - self.cfg.max_position_difference_rad
        upper = dof_pos + self.cfg.max_position_difference_rad
        max_target_position = torch.minimum(torch.maximum(target, lower), upper)
        delta = torch.clamp(max_target_position - req, min=-speed_limit, max=speed_limit)
        return req + delta

    def inference(self) -> torch.Tensor:
        self._transition()

        if self.state == "recovery":
            dof_targets = self._recover_target()
        elif self.state == "stand_up":
            obs = self._compute_rl_observation()
            with torch.no_grad():
                action = self._model(obs).squeeze(0)
                action = torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)

            dof_targets = self.robot.default_joint_pos.clone()
            dof_targets[self.real2sim_joint_map] = self.offset_joint_pos + self.cfg.action_scale * action
        elif self.state == "break_up":
            dof_targets = self._break_up_target()
        else:
            # Return to kick-style idle posture after get-up is completed.
            dof_targets = self.done_joint_pos.clone()

        self.last_output = dof_targets.clone()
        self.last_request = dof_targets[self.real2sim_joint_map] - self.offset_joint_pos
        return dof_targets


@configclass
class StandupPolicyCfg(PolicyCfg):
    constructor = StandupPolicy
    checkpoint_path: str = MISSING  # type: ignore

    action_scale: float = 1.0
    obs_dof_vel_scale: float = 0.1
    clip_actions: float = 3.0
    policy_dt: float = 0.02

    max_try_counter: int = 3
    speed_factor: float = 1.5
    use_break_up: bool = False
    earliest_done_ratio: float = 0.95
    min_stand_height_when_done: float = 0.43
    front_total_time_s: float = 5.25
    back_total_time_s: float = 6.0
    break_up_wait_s: float = 2.0
    break_up_joint_speed_rad_s: float = _deg(60.0)
    break_up_head_angle_rad: float = _deg(20.0)
    max_position_difference_rad: float = _deg(2.0)

    recover_front_roll_threshold_rad: float = _deg(40.0)
    recover_front_pitch_threshold_rad: float = _deg(30.0)
    recover_back_roll_threshold_rad: float = _deg(40.0)
    recover_back_pitch_threshold_rad: float = _deg(30.0)

    enable_safety_fallback: bool = False

    default_joint_pos: list[float] = [
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    ]
    # Kick-style finish posture (arms opened to +/- 1.35 shoulder roll).
    done_joint_pos: list[float] = [
        0.0, 0.0,
        0.0, -1.35, 0.0, 0.0,
        0.0, 1.35, 0.0, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    ]

    fall_angles: list[float] = [
        0.0, 0.0,
        0.0, _deg(-83.0), _deg(90.0), 0.0,
        0.0, _deg(83.0), _deg(90.0), 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    # (joint_pose_22, duration_s)
    recover_front: list[tuple[list[float], float]] = [
        ([
            0.0, _deg(-20.0),
            0.0, _deg(-40.0), 0.0, 0.0,
            0.0, _deg(40.0), 0.0, 0.0,
            _deg(-50.0), _deg(20.0), 0.0, 0.0, 0.0, 0.0,
            _deg(-50.0), _deg(-20.0), 0.0, 0.0, 0.0, 0.0,
        ], 0.5),
        ([
            0.0, _deg(-20.0),
            0.0, _deg(-83.0), 0.0, 0.0,
            0.0, _deg(83.0), 0.0, 0.0,
            _deg(-40.0), _deg(10.0), 0.0, _deg(20.0), 0.0, 0.0,
            _deg(-40.0), _deg(-10.0), 0.0, _deg(20.0), 0.0, 0.0,
        ], 0.5),
    ]
    recover_back: list[tuple[list[float], float]] = [
        ([
            0.0, _deg(20.0),
            0.0, _deg(-40.0), 0.0, 0.0,
            0.0, _deg(40.0), 0.0, 0.0,
            _deg(-60.0), _deg(20.0), 0.0, 0.0, 0.0, 0.0,
            _deg(60.0), _deg(-20.0), 0.0, 0.0, 0.0, 0.0,
        ], 0.5),
        ([
            0.0, _deg(20.0),
            0.0, _deg(-83.0), 0.0, 0.0,
            0.0, _deg(83.0), 0.0, 0.0,
            _deg(-40.0), _deg(10.0), 0.0, _deg(20.0), 0.0, 0.0,
            _deg(40.0), _deg(-10.0), 0.0, _deg(20.0), 0.0, 0.0,
        ], 0.5),
    ]
    recover_normal: list[tuple[list[float], float]] = [
        ([
            0.0, 0.0,
            0.0, _deg(-83.0), 0.0, 0.0,
            0.0, _deg(83.0), 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], 1.0),
    ]

    # (x_min, x_max, time_s, y_min, y_max), radians/s
    front_info: list[tuple[float, float, float, float, float]] = [
        (_deg(-40.0), _deg(40.0), 1.0, _deg(70.0), _deg(130.0)),
        (_deg(-40.0), _deg(40.0), 2.0, _deg(70.0), _deg(140.0)),
        (_deg(-40.0), _deg(40.0), 2.5, _deg(70.0), _deg(120.0)),
        (_deg(-40.0), _deg(40.0), 3.5, _deg(50.0), _deg(110.0)),
        (_deg(-30.0), _deg(30.0), 4.0, _deg(20.0), _deg(70.0)),
        (_deg(-20.0), _deg(20.0), 4.25, _deg(0.0), _deg(60.0)),
        (_deg(-20.0), _deg(20.0), 5.25, _deg(-30.0), _deg(45.0)),
    ]
    back_info: list[tuple[float, float, float, float, float]] = [
        (_deg(-40.0), _deg(40.0), 1.0, _deg(-110.0), _deg(-70.0)),
        (_deg(-40.0), _deg(40.0), 2.5, _deg(-110.0), _deg(-70.0)),
        (_deg(-40.0), _deg(40.0), 3.0, _deg(-110.0), _deg(-70.0)),
        (_deg(-40.0), _deg(40.0), 3.5, _deg(-150.0), _deg(-90.0)),
        (_deg(-40.0), _deg(40.0), 4.25, _deg(-150.0), _deg(-100.0)),
        (_deg(-30.0), _deg(30.0), 5.0, _deg(-120.0), _deg(-50.0)),
        (_deg(-20.0), _deg(20.0), 6.0, _deg(-30.0), _deg(30.0)),
    ]

    policy_joint_names: list[str] = MISSING  # type: ignore
    lower_body_joint_names: list[str] = MISSING  # type: ignore


@configclass
class K1StandupControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        default_joint_pos=[
            0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
        ],
        joint_stiffness=[
            10.0, 10.0,
            40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0,
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
        lower_body_joint_names=[
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
