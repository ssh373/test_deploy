"""Deploy the K1 GoTo policy with the same actor contract used in training."""

from __future__ import annotations

import math
import os
import time
from dataclasses import MISSING

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    PolicyCfg,
    PrepareStateCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass


# This is Isaac Lab's resolved articulation order for LEGS.  It is also the
# action/observation order in the exported policy.
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
    0.2, -1.3, 0.0, -0.5,
    0.2, 1.3, 0.0, 0.5,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
]

# Training fixes the 10 head/arm joints. MuJoCo models them as hinges, so use
# the robot's prepare gains to approximate that fixed zero pose.
JOINT_STIFFNESS = [
    40.0, 40.0,
    40.0, 50.0, 20.0, 20.0,
    40.0, 50.0, 20.0, 20.0,
] + [100.0] * 4 + [50.0] * 2 + [100.0] * 4 + [50.0] * 2
JOINT_DAMPING = [
    1.5, 1.5,
    0.5, 1.5, 0.2, 0.2,
    0.5, 1.5, 0.2, 0.2,
] + [2.0] * 4 + [1.0] * 2 + [2.0] * 4 + [1.0] * 2
EFFORT_LIMIT = [
    6.0, 6.0,
    14.0, 14.0, 14.0, 14.0,
    14.0, 14.0, 14.0, 14.0,
    68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
    68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
]

PREPARE_STIFFNESS = K1_CFG.prepare_state.stiffness
PREPARE_DAMPING = K1_CFG.prepare_state.damping


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Return yaw from a scalar-first (w, x, y, z) quaternion."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class K1GoToPolicy(Policy):
    """Recurrent pose-goal policy.

    The controller's three-number command is interpreted as a new body-relative
    pose goal ``dx dy dyaw``.  It is latched in the world frame, so the command
    naturally approaches zero as MuJoCo's robot moves.
    """

    def __init__(self, cfg: "K1GoToPolicyCfg", controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.controller = controller
        self.robot = controller.robot
        self.device = torch.device(cfg.device)

        path = cfg.checkpoint_path
        if not os.path.isabs(path):
            path = os.path.join(self.task_path, path)
        try:
            self._model: torch.jit.ScriptModule = torch.jit.load(path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load recurrent TorchScript policy: {path}\n"
                "Export it with train/scripts/rsl_rl/play.py and copy policy.pt here."
            ) from exc
        self._model.to(self.device).eval()
        self.robot.data.to(self.device)

        self.policy_joint_idx = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in cfg.policy_joint_names],
            dtype=torch.long, device=self.device,
        )
        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)
        self.action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(len(cfg.policy_joint_names), device=self.device)
        self.goal_pose_w = torch.zeros(3, device=self.device)
        self._last_input: tuple[float, float, float] | None = None
        self._use_robot_odometry = controller.__class__.__name__ == "BoosterRobotController"
        self._odom_node = None
        self._odom_context = None
        self._odom_xyyaw: tuple[float, float, float] | None = None
        self._odom_received_at = 0.0
        self._odom_receive_count = 0
        self._pending_relative_goal: tuple[float, float, float] | None = None
        self._goal_event_sequence = 0
        self._handled_goal_sequence = 0
        self._last_pose_xyyaw = (0.0, 0.0, 0.0)
        self._last_goal_command = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        )
        if self._use_robot_odometry:
            self._init_robot_odometry()

    def _init_robot_odometry(self) -> None:
        """Create a task-local ROS 2 subscriber in the inference process."""
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.node import Node
            from booster_interface.msg import Odometer
            from geometry_msgs.msg import Pose2D
        except ImportError as exc:
            raise RuntimeError(
                "Real GoTo deployment requires rclpy and "
                "booster_interface/msg/Odometer plus geometry_msgs/msg/Pose2D."
            ) from exc

        # The inference worker is a separate process. Use a private context so
        # this task-local node does not share the portal's default ROS context.
        self._odom_context = Context()
        rclpy.init(context=self._odom_context)
        self._odom_node = Node("k1_goto_odometry", context=self._odom_context)

        def callback(msg) -> None:
            self._odom_xyyaw = (float(msg.x), float(msg.y), float(msg.theta))
            self._odom_received_at = time.monotonic()
            self._odom_receive_count += 1

        self._odom_subscription = self._odom_node.create_subscription(
            Odometer, self.cfg.odometry_topic, callback, 10
        )

        def goal_callback(msg) -> None:
            # Each callback is a new event. Numerically identical consecutive
            # Pose2D messages therefore create distinct relative goals.
            self._pending_relative_goal = (
                float(msg.x), float(msg.y), float(msg.theta)
            )
            self._goal_event_sequence += 1
            print(
                f"Received relative goal #{self._goal_event_sequence}: "
                f"x={msg.x:.3f}, y={msg.y:.3f}, theta={msg.theta:.3f}"
            )

        self._goal_subscription = self._odom_node.create_subscription(
            Pose2D, self.cfg.relative_goal_topic, goal_callback, 10
        )

    def _update_robot_odometry(self, timeout_sec: float = 0.0) -> None:
        if not self._use_robot_odometry:
            return
        import rclpy
        rclpy.spin_once(
            self._odom_node,
            timeout_sec=timeout_sec,
        )

    def _pose_xyyaw(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return x, y, yaw from MuJoCo state or the real odometry topic."""
        if self._use_robot_odometry:
            self._update_robot_odometry()
            if self._odom_xyyaw is None:
                raise RuntimeError(
                    f"No message received from {self.cfg.odometry_topic}."
                )
            age = time.monotonic() - self._odom_received_at
            if age > self.cfg.odometry_timeout_s:
                self.controller.stop()
                raise RuntimeError(
                    f"Stale {self.cfg.odometry_topic}: last message is {age:.3f}s old."
                )
            x, y, yaw = self._odom_xyyaw
            self._last_pose_xyyaw = (x, y, yaw)
            return (
                torch.tensor(x, dtype=torch.float32, device=self.device),
                torch.tensor(y, dtype=torch.float32, device=self.device),
                torch.tensor(yaw, dtype=torch.float32, device=self.device),
            )
        position = self.robot.data.root_pos_w
        yaw = _yaw_from_quat(self.robot.data.root_quat_w)
        self._last_pose_xyyaw = (
            float(position[0].item()), float(position[1].item()), float(yaw.item())
        )
        return position[0], position[1], yaw

    def _reset_recurrent_state(self) -> None:
        reset = getattr(self._model, "reset", None)
        if reset is None:
            return
        try:
            reset()
        except (RuntimeError, TypeError):
            # Some RSL-RL exporter versions expose reset(done_mask).
            reset(torch.ones(1, dtype=torch.bool, device=self.device))

    def reset(self) -> None:
        self.last_action.zero_()
        self._last_input = None
        self._reset_recurrent_state()
        if self._use_robot_odometry:
            # Fail before commanding a pose if real odometry is unavailable.
            deadline = time.monotonic() + self.cfg.odometry_startup_timeout_s
            while self._odom_xyyaw is None and time.monotonic() < deadline:
                self._update_robot_odometry(timeout_sec=0.1)
            if self._odom_xyyaw is None:
                self.controller.stop()
                raise RuntimeError(
                    f"Timed out waiting for {self.cfg.odometry_topic}."
                )
        self._latch_goal(force=True)

    def _command_input(self) -> tuple[float, float, float]:
        if self._use_robot_odometry:
            # Real-robot GoTo commands are one-shot ROS events, not the
            # continuously refreshed keyboard/velocity command channel.
            return self._pending_relative_goal or self.cfg.default_goal
        cmd = self.controller.vel_command
        if cmd is None:
            return self.cfg.default_goal
        return (
            max(-cmd.vx_max, min(cmd.vx_max, cmd.lin_vel_x)),
            max(-cmd.vy_max, min(cmd.vy_max, cmd.lin_vel_y)),
            max(-cmd.vyaw_max, min(cmd.vyaw_max, cmd.ang_vel_yaw)),
        )

    def _latch_goal(self, force: bool = False) -> None:
        event_sequence = self._goal_event_sequence
        if self._use_robot_odometry:
            # Process both odometry and goal callbacks before checking whether
            # a new one-shot command arrived.
            self._update_robot_odometry()
            event_sequence = self._goal_event_sequence
            if not force and self._goal_event_sequence == self._handled_goal_sequence:
                return
        command = self._command_input()
        if not self._use_robot_odometry and not force and command == self._last_input:
            return
        x, y, yaw = self._pose_xyyaw()
        dx, dy, dyaw = command
        c, s = torch.cos(yaw), torch.sin(yaw)
        self.goal_pose_w[0] = x + c * dx - s * dy
        self.goal_pose_w[1] = y + s * dx + c * dy
        self.goal_pose_w[2] = lab_math.wrap_to_pi(yaw + dyaw)
        self._last_input = command
        if self._use_robot_odometry:
            # A second callback can arrive while fetching the pose below. Mark
            # only the event whose command was actually latched as handled.
            self._handled_goal_sequence = event_sequence

    def _goal_command(self) -> torch.Tensor:
        self._latch_goal()
        x, y, yaw = self._pose_xyyaw()
        delta = self.goal_pose_w[:2] - torch.stack((x, y))
        c, s = torch.cos(yaw), torch.sin(yaw)
        dx = c * delta[0] + s * delta[1]
        dy = -s * delta[0] + c * delta[1]
        dyaw = lab_math.wrap_to_pi(self.goal_pose_w[2] - yaw)
        return torch.stack((dx, dy, torch.sin(dyaw), torch.cos(dyaw)))

    def compute_observation(self) -> torch.Tensor:
        quat = self.robot.data.root_quat_w
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        projected_gravity = lab_math.quat_apply_inverse(quat, gravity)
        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print("\nFalling detected, stopping GoTo policy.")
            self.controller.stop()

        idx = self.policy_joint_idx
        goal_command = self._goal_command()
        self._last_goal_command.copy_(goal_command.detach())
        obs = torch.cat((
            self.robot.data.root_ang_vel_b * 0.25,
            projected_gravity,
            self.robot.data.joint_pos[idx] - self.default_joint_pos[idx],
            self.robot.data.joint_vel[idx] * 0.05,
            self.last_action,
            goal_command,
        ))
        if obs.numel() != self.cfg.obs_dim:
            raise RuntimeError(f"Expected GoTo obs dim {self.cfg.obs_dim}, got {obs.numel()}")
        return obs.unsqueeze(0)

    def inference(self) -> torch.Tensor:
        with torch.no_grad():
            output = self._model(self.compute_observation())
            action = output[0] if isinstance(output, tuple) else output
            action = action.reshape(-1)
        if action.numel() != len(self.cfg.policy_joint_names):
            raise RuntimeError(f"Expected 12 actions, got {action.numel()}")
        if self.controller._step_count % 25 == 0:
            x, y, yaw = self._last_pose_xyyaw
            odom_age = (
                time.monotonic() - self._odom_received_at
                if self._use_robot_odometry and self._odom_received_at > 0.0
                else 0.0
            )
            print(
                f"raw_odom=({x:.6f}, {y:.6f}, {yaw:.6f})",
                f"odom_rx={self._odom_receive_count}",
                f"odom_age={odom_age:.3f}s",
                "remaining_goal=", self._last_goal_command.cpu().numpy(),
                "raw_action=", action.detach().cpu().numpy(),
                flush=True,
            )
        # GoTo training uses RSL-RL's default (no action clipping). Clipping
        # only in deployment destroys the relative magnitudes of large,
        # coordinated leg actions.
        if self.cfg.clip_actions is not None:
            action = torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)
        self.last_action.copy_(action)
        targets = self.default_joint_pos.clone()
        targets[self.policy_joint_idx] += action * self.action_scale
        return targets


@configclass
class K1GoToPolicyCfg(PolicyCfg):
    constructor = K1GoToPolicy
    checkpoint_path: str = MISSING  # type: ignore
    # ang_vel(3) + gravity(3) + joint_pos(12) + joint_vel(12)
    # + previous_action(12) + goal(4)
    obs_dim: int = 46
    clip_actions: float | None = None
    enable_safety_fallback: bool = True
    policy_joint_names: list[str] = POLICY_JOINT_NAMES
    default_goal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    odometry_topic: str = "/odometer_state"
    relative_goal_topic: str = "/goto/relative_goal"
    odometry_timeout_s: float = 0.5
    odometry_startup_timeout_s: float = 3.0
    # 0.25 * effort_limit / stiffness, in POLICY_JOINT_NAMES order.
    action_scale: tuple[float, ...] = (
        0.17, 0.17, 0.19, 0.19, 0.09575, 0.09575,
        0.28, 0.28, 0.1915, 0.1915, 0.1915, 0.1915,
    )


@configclass
class K1GoToControllerCfg(ControllerCfg):
    policy_dt: float = 0.02
    robot = K1_CFG.replace(
        default_joint_pos=DEFAULT_JOINT_POS,
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=JOINT_DAMPING,
        effort_limit=EFFORT_LIMIT,
        prepare_state=PrepareStateCfg(
            stiffness=PREPARE_STIFFNESS,
            damping=PREPARE_DAMPING,
            joint_pos=DEFAULT_JOINT_POS,
        ),
    )
    # Existing controller input is reused; values mean dx [m], dy [m], dyaw [rad].
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=2.0, vy_max=1.5, vyaw_max=math.pi,
    )
    policy: K1GoToPolicyCfg = K1GoToPolicyCfg(policy_joint_names=POLICY_JOINT_NAMES)
    mujoco: MujocoControllerCfg = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.58],
        init_quat=[1.0, 0.0, 0.0, 0.0],
        decimation=4,
    )
