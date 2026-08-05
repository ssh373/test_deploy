"""Deploy the 49-observation Isaac Lab K1 kick policy in MuJoCo."""

from __future__ import annotations

import math
import os
from dataclasses import MISSING

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    PolicyCfg,
    PrepareStateCfg,
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass


# Isaac Lab's resolved regex order, shared by observations and actions.
POLICY_JOINT_NAMES = [
    "Left_Hip_Pitch", "Right_Hip_Pitch",
    "Left_Hip_Roll", "Right_Hip_Roll",
    "Left_Hip_Yaw", "Right_Hip_Yaw",
    "Left_Knee_Pitch", "Right_Knee_Pitch",
    "Left_Ankle_Pitch", "Right_Ankle_Pitch",
    "Left_Ankle_Roll", "Right_Ankle_Roll",
]

DEFAULT_JOINT_POS = [
    0.0, 0.0,
    0.0, -1.35, 0.0, 0.0,
    0.0, 1.35, 0.0, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
]
JOINT_STIFFNESS = [
    40.0, 40.0,
    40.0, 50.0, 20.0, 20.0,
    40.0, 50.0, 20.0, 20.0,
    80.0, 80.0, 80.0, 80.0, 30.0, 30.0,
    80.0, 80.0, 80.0, 80.0, 30.0, 30.0,
]
JOINT_DAMPING = [
    1.5, 1.5,
    0.5, 1.5, 0.2, 0.2,
    0.5, 1.5, 0.2, 0.2,
] + [2.0] * 12
EFFORT_LIMIT = [
    6.0, 6.0,
    14.0, 14.0, 14.0, 14.0,
    14.0, 14.0, 14.0, 14.0,
    68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
    68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
]

_MJCF_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "assets", "K1_22dof_kick_ball.xml"
))


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class K1IsaacLabKickPolicy(Policy):
    """Exact actor-side contract of Booster-K1-Kick_001-v0."""

    def __init__(self, cfg: "K1IsaacLabKickPolicyCfg", controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.controller = controller
        self.robot = controller.robot
        self.device = torch.device(cfg.device)

        checkpoint = cfg.checkpoint_path
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(self.task_path, checkpoint)
        try:
            self.model: torch.jit.ScriptModule = torch.jit.load(checkpoint, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load kick TorchScript policy: {checkpoint}\n"
                "Export policy.pt with the Isaac Lab play script and copy it to "
                "tasks/kick_isaaclab/models/k1_kick_jit.pt."
            ) from exc
        self.model.to(self.device).eval()
        self.robot.data.to(self.device)

        self.joint_idx = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in cfg.policy_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        print(f"[kick deploy] policy_joint_names={cfg.policy_joint_names}", flush=True)
        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)
        self.action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(12, dtype=torch.float32, device=self.device)
        self.target_w = torch.zeros(2, dtype=torch.float32, device=self.device)
        self._ball_body_id: int | None = None
        self._diagnostic_steps = 0
        self._prepare_logged = False

    def _reset_model(self) -> None:
        reset = getattr(self.model, "reset", None)
        if reset is None:
            return
        try:
            reset()
        except (RuntimeError, TypeError):
            reset(torch.ones(1, dtype=torch.bool, device=self.device))

    def reset(self) -> None:
        # In custom mode the Booster RotateHead API does not actuate the neck
        # directly.  The portal captures that RPC and this controller applies
        # the same target through the low-level neck joints.
        if not hasattr(self.controller, "mj_model"):
            self.controller.pass_through_joint_idx = []
            self.controller.head_track_yaw_idx = self.robot.cfg.joint_names.index("AAHead_yaw")
            self.controller.head_track_pitch_idx = self.robot.cfg.joint_names.index("Head_pitch")
            self.controller.head_track_from_loco_api = True
            self.controller.head_track_from_ball = False
        self.last_action.zero_()
        self._diagnostic_steps = 0
        self._prepare_logged = False
        self._reset_model()
        self._reset_ball_and_target()

    def _reset_ball_and_target(self) -> None:
        root_pos = self.robot.data.root_pos_w.to(self.device)
        yaw = _yaw_from_quat(self.robot.data.root_quat_w.to(self.device))
        c, s = torch.cos(yaw), torch.sin(yaw)
        angle = math.radians(self.cfg.target_angle_deg)
        local_angle = yaw + angle
        self.target_w[0] = root_pos[0] + self.cfg.target_distance * torch.cos(local_angle)
        self.target_w[1] = root_pos[1] + self.cfg.target_distance * torch.sin(local_angle)

        # Real-robot deployment receives the ball in robot-base coordinates
        # from /booster_vision/ball.  Only MuJoCo needs a simulated ball body.
        if not hasattr(self.controller, "mj_model"):
            return
        import mujoco

        model = self.controller.mj_model
        data = self.controller.mj_data
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.cfg.ball_joint_name)
        self._ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.ball_body_name)
        if joint_id < 0 or self._ball_body_id < 0:
            raise RuntimeError("Kick MJCF must contain ball and ball_freejoint.")

        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.cfg.ball_geom_name)
        if geom_id < 0:
            raise RuntimeError(f"Kick MJCF has no geom named {self.cfg.ball_geom_name!r}.")
        model.geom_size[geom_id, 0] = self.cfg.ball_radius
        model.geom_friction[geom_id, 0] = self.cfg.ball_sliding_friction
        model.body_mass[self._ball_body_id] = self.cfg.ball_mass
        inertia = 0.4 * self.cfg.ball_mass * self.cfg.ball_radius ** 2
        model.body_inertia[self._ball_body_id, :] = inertia

        bx, by = self.cfg.ball_spawn_rel_xy
        ball_xy = torch.stack((root_pos[0] + c * bx - s * by, root_pos[1] + s * bx + c * by))

        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr:qpos_adr + 3] = [float(ball_xy[0]), float(ball_xy[1]), self.cfg.ball_height]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[qvel_adr:qvel_adr + 6] = 0.0

        mujoco.mj_forward(model, data)

    def _ball_relative_xy(self, yaw: torch.Tensor) -> torch.Tensor:
        if self._ball_body_id is None:
            if not hasattr(self.controller, "ball_rel_xy"):
                return torch.tensor(self.cfg.missing_ball_rel_xy, dtype=torch.float32, device=self.device)
            return self.controller.ball_rel_xy.to(self.device).clamp(-3.0, 3.0)
        ball_w = torch.as_tensor(
            self.controller.mj_data.xpos[self._ball_body_id, :2],
            dtype=torch.float32,
            device=self.device,
        )
        delta = ball_w - self.robot.data.root_pos_w[:2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.stack((c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1])).clamp(-3.0, 3.0)

    def _target_relative_xy(self, yaw: torch.Tensor) -> torch.Tensor:
        delta = self.target_w - self.robot.data.root_pos_w[:2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        relative = torch.stack((c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]))
        # Isaac Lab observation terms apply clipping before scaling.  For the
        # 4 m target this must produce 2.0 * 0.25 = 0.5, not 1.0.  Reversing
        # these operations puts target-X over 100 normalizer standard
        # deviations out of distribution and makes the actor saturate.
        return relative.clamp(-2.0, 2.0) * 0.25

    def compute_observation(self) -> torch.Tensor:
        quat = self.robot.data.root_quat_w
        yaw = _yaw_from_quat(quat)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        gravity_b = lab_math.quat_apply_inverse(quat, gravity_w)
        if self.cfg.enable_safety_fallback and gravity_b[2] > -0.5:
            print("Falling detected; stopping kick policy.", flush=True)
            self.controller.stop()

        idx = self.joint_idx
        if self._ball_body_id is None:
            # This checkpoint was trained only with [visible, age, confidence]
            # == [1, 0, 1].  Confidence/age are used by the external safety
            # gate, but must not be fed numerically until dropout fine-tuning.
            perception = torch.tensor([1.0, 0.0, 1.0], device=self.device)
        else:
            perception = torch.tensor([1.0, 0.0, 1.0], device=self.device)
        obs = torch.cat((
            gravity_b,
            self.robot.data.root_ang_vel_b,
            self._ball_relative_xy(yaw),
            self._target_relative_xy(yaw),
            perception,
            self.robot.data.joint_pos[idx] - self.default_joint_pos[idx],
            self.robot.data.joint_vel[idx] * 0.1,
            self.last_action,
        ))
        if obs.numel() != self.cfg.obs_dim:
            raise RuntimeError(f"Expected {self.cfg.obs_dim} kick observations, got {obs.numel()}.")
        return obs.unsqueeze(0)

    def inference(self) -> torch.Tensor:
        if self._ball_body_id is None and float(getattr(self.controller, "ball_visible", 0.0)) < 0.5:
            # The current checkpoint never trained on stale/missing vision.
            # Hold the safe start pose rather than act on a phantom coordinate.
            self.last_action.zero_()
            return self.default_joint_pos.clone()
        if self.cfg.hold_default_pose:
            if self._diagnostic_steps == 0:
                print(
                    "[kick diagnostic] Holding the default pose; policy actions are disabled.",
                    flush=True,
                )
            self._diagnostic_steps += 1
            return self.default_joint_pos.clone()

        elapsed_s = float(getattr(self.controller, "_elapsed_s", 0.0))
        if elapsed_s <= self.cfg.startup_hold_time_s:
            if not self._prepare_logged:
                print(
                    f"[kick prepare] Holding the training default pose for "
                    f"{self.cfg.startup_hold_time_s:.2f}s before policy control.",
                    flush=True,
                )
                self._prepare_logged = True
            return self.default_joint_pos.clone()

        with torch.no_grad():
            output = self.model(self.compute_observation())
            action = output[0] if isinstance(output, tuple) else output
            action = action.reshape(-1)
        if action.numel() != 12 or not torch.isfinite(action).all():
            self.controller.stop()
            raise RuntimeError(f"Invalid kick action shape/value: {tuple(action.shape)}")
        if self.cfg.clip_actions is not None:
            action = action.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        self.last_action.copy_(action)
        targets = self.default_joint_pos.clone()
        targets[self.joint_idx] += action * self.action_scale
        if self._diagnostic_steps < self.cfg.diagnostic_print_steps:
            current = self.robot.data.joint_pos[self.joint_idx]
            torque = self.robot.data.feedback_torque[self.joint_idx]
            print(
                f"[kick step {self._diagnostic_steps}]\n"
                f"  action={action.detach().cpu().tolist()}\n"
                f"  current={current.detach().cpu().tolist()}\n"
                f"  target={targets[self.joint_idx].detach().cpu().tolist()}\n"
                f"  torque={torque.detach().cpu().tolist()}",
                flush=True,
            )
        self._diagnostic_steps += 1
        return targets


@configclass
class K1IsaacLabKickPolicyCfg(PolicyCfg):
    constructor = K1IsaacLabKickPolicy
    checkpoint_path: str = MISSING  # type: ignore
    obs_dim: int = 49
    clip_actions: float | None = None
    hold_default_pose: bool = False
    startup_hold_time_s: float = 0.4
    diagnostic_print_steps: int = 10
    enable_safety_fallback: bool = True
    policy_joint_names: list[str] = POLICY_JOINT_NAMES
    action_scale: tuple[float, ...] = (
        0.2125, 0.2125,
        0.2375, 0.2375,
        0.1196875, 0.1196875,
        0.35, 0.35,
        0.3191666667, 0.3191666667,
        0.3191666667, 0.3191666667,
    )
    ball_body_name: str = "ball"
    ball_geom_name: str = "ball_geom"
    ball_joint_name: str = "ball_freejoint"
    ball_spawn_rel_xy: tuple[float, float] = (0.275, 0.0)
    ball_radius: float = 0.103
    ball_mass: float = 0.37
    ball_sliding_friction: float = 0.3
    ball_height: float = 0.105
    target_distance: float = 4.0
    target_angle_deg: float = 0.0
    missing_ball_rel_xy: tuple[float, float] = (0.275, 0.0)
    ball_timeout_s: float = 0.3
    ball_confidence_threshold: float = 0.4


@configclass
class K1IsaacLabKickControllerCfg(ControllerCfg):
    policy_dt: float = 0.02
    head_control_from_loco_api: bool = True
    fall_protection_enabled: bool = True
    fall_roll_pitch_threshold_rad: float = math.radians(30.0)
    fall_trigger_duration_s: float = 0.10
    robot = K1_CFG.replace(
        mjcf_path=_MJCF_PATH,
        default_joint_pos=DEFAULT_JOINT_POS,
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=JOINT_DAMPING,
        effort_limit=EFFORT_LIMIT,
        prepare_state=PrepareStateCfg(
            stiffness=K1_CFG.prepare_state.stiffness,
            damping=K1_CFG.prepare_state.damping,
            joint_pos=DEFAULT_JOINT_POS,
        ),
    )
    policy: K1IsaacLabKickPolicyCfg = K1IsaacLabKickPolicyCfg()
    mujoco: MujocoControllerCfg = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.57],
        init_quat=[1.0, 0.0, 0.0, 0.0],
        decimation=4,
    )
