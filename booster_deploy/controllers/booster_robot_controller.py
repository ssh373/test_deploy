from __future__ import annotations
import logging
import json
import math
import signal
import time
import threading
import multiprocessing as mp
from multiprocessing import synchronize

import numpy as np
import torch

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from booster_interface.msg import LowState, LowCmd, MotorCmd
try:
    from booster_msgs.msg import RpcReqMsg
except Exception:
    RpcReqMsg = None
try:
    from vision_interface.msg import Ball as VisionBall
except Exception:
    VisionBall = None

from booster_robotics_sdk_python import (  # type: ignore
    B1LocoClient,
    RobotMode,
)

from .controller_cfg import ControllerCfg
from .base_controller import BaseController, BoosterRobot
from ..utils.synced_array import SyncedArray
from ..utils.metrics import SyncedMetrics
from ..utils.isaaclab import math as lab_math
from ..utils.remote_control_service import RemoteControlService


logger = logging.getLogger("booster_deploy")
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


class CountTimer:
    def __init__(self, dt: float = 0.002, use_sim_time: bool = False):
        self.dt = dt
        # Use multiprocessing.Value for inter-process communication
        self.counter = mp.Value('L', 0)
        self.use_sim_time = use_sim_time

    def tick_timer_if_sim(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                self.counter.value += 1

    def get_time(self):
        if self.use_sim_time:
            with self.counter.get_lock():
                return self.counter.value * self.dt
        else:
            return time.perf_counter()


class BoosterRobotPortal:
    synced_state: SyncedArray
    synced_command: SyncedArray
    synced_action: SyncedArray
    exit_event: synchronize.Event

    def __init__(self, cfg: ControllerCfg, use_sim_time: bool = False) -> None:
        self.cfg = cfg

        self.robot = BoosterRobot(cfg.robot)
        missing_ball_rel_xy = list(
            getattr(self.cfg.policy, "missing_ball_rel_xy", [0.5, 0.0])
        )
        if len(missing_ball_rel_xy) < 2:
            missing_ball_rel_xy = [0.5, 0.0]
        self._ball_default_rel_xy = np.array(
            missing_ball_rel_xy[:2],
            dtype=np.float32,
        )
        self._ball_rel_xy = self._ball_default_rel_xy.copy()
        self._ball_last_seen_time = 0.0
        self._ball_valid = False
        self._ball_timeout_s = 0.3
        self._head_cmd_pitch = 0.0
        self._head_cmd_yaw = 0.0
        self._head_cmd_last_seen_time = 0.0
        self._head_cmd_valid = False
        self._head_cmd_timeout_s = 2.0

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.remoteControlService = RemoteControlService()
        # Use multiprocessing.Event for inter-process communication
        self.exit_event = mp.Event()
        self.is_running = True
        self.timer = CountTimer(
            self.cfg.booster.low_state_dt, use_sim_time=use_sim_time)

        def signal_handler(sig, frame):
            if mp.current_process().name == "MainProcess":
                print("\nKeyboard interrupt received. Shutting down...")
            self.exit_event.set()

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)

        self._init_synced_buffer()
        self._init_metrics()

        self._cleanup_done = False
        self.inference_process = None  # Inference process reference
        self.low_cmd_publisher: rclpy.publisher.Publisher = None
        self.low_state_thread = None
        self.low_cmd_process: mp.Process | None = None

        rclpy.init()
        # Initialize communication. Callbacks may start immediately and
        # reference `is_running` and `exit_event`, so ensure those are set.
        self._init_communication()

    def _init_synced_buffer(self):
        action_dtype = np.dtype(
            [
                ("dof_target", float, (self.robot.num_joints,)),
                ("stiffness", float, (self.robot.num_joints,)),
                ("damping", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_action = SyncedArray(
            "action",
            shape=(1,),
            dtype=action_dtype,
        )
        self._action_buf = np.ndarray((1,), dtype=action_dtype)

        state_dtype = np.dtype(
            [
                ("root_rpy_w", float, (3,)),
                ("root_ang_vel_b", float, (3,)),
                ("root_pos_w", float, (3,)),
                ("root_lin_vel_w", float, (3,)),
                ("joint_pos", float, (self.robot.num_joints,)),
                ("joint_vel", float, (self.robot.num_joints,)),
                ("feedback_torque", float, (self.robot.num_joints,)),
            ]
        )
        self.synced_state = SyncedArray(
            "state",
            shape=(1,),
            dtype=state_dtype
        )
        self._state_buf = np.zeros((1,), dtype=state_dtype)

        command_dtype = np.dtype(
            [
                ("vx", float),
                ("vy", float),
                ("vyaw", float),
                ("ball_rel_x", float),
                ("ball_rel_y", float),
                ("head_pitch_cmd", float),
                ("head_yaw_cmd", float),
                ("head_cmd_active", float),
            ]
        )
        self.synced_command = SyncedArray(
            "command",
            shape=(1,),
            dtype=command_dtype,
        )

    def _init_metrics(self):
        # initialize cross-process synced metrics
        max_events = self.cfg.booster.metrics_max_events
        self.metrics = {
            "low_state_handler": SyncedMetrics(
                "low_state_handler", max_events=max_events
            ),
            "policy_step": SyncedMetrics(
                "policy_step", max_events=max_events
            ),
        }

    def _init_communication(self) -> None:
        try:
            self.client = B1LocoClient()
            self.create_low_cmd_publisher("booster_deploy_low_cmd_pub")
            self._start_low_state_subscription()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _start_low_state_subscription(self) -> None:
        """Start ROS 2 subscription loop on a dedicated thread.

        The subscription is run on a dedicated thread and spins a
        SingleThreadedExecutor for the `/low_state` topic.
        """

        def low_state_service_executor():
            self.logger.info("Low state subscription started")
            low_state_node = rclpy.create_node("booster_deploy_low_state_sub")
            low_state_node.create_subscription(
                LowState,
                "/low_state",
                self._low_state_handler,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )
            if VisionBall is not None:
                low_state_node.create_subscription(
                    VisionBall,
                    # "booster_vision/ball",
                    "brain/ball_memory",
                    self._vision_ball_handler,
                    QoSProfile(
                        depth=1,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST,
                    ),
                )
                self.logger.info(
                    "Vision ball subscriber started: topic='booster_vision/ball', msg='vision_interface/msg/Ball'"
                )
            else:
                self.logger.warning(
                    "vision_interface.msg.Ball import failed. Using default ball relative xy."
                )
            if RpcReqMsg is not None:
                low_state_node.create_subscription(
                    RpcReqMsg,
                    "LocoApiTopicReq",
                    self._loco_api_req_handler,
                    QoSProfile(
                        depth=10,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST,
                    ),
                )
                self.logger.info("Subscribed to LocoApiTopicReq for moveHead integration.")

            executor = SingleThreadedExecutor()
            executor.add_node(low_state_node)

            try:
                # loop: check exit_event and rclpy.ok()
                while rclpy.ok() and not self.exit_event.is_set():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass
            except Exception as exc:
                # Suppress RCLError if we are shutting down
                is_rcl_error = "RCLError" in type(exc).__name__
                is_shutting_down = self.exit_event.is_set() or not rclpy.ok()

                if is_rcl_error and is_shutting_down:
                    pass
                else:
                    self.logger.error(
                        "Low state subscription executor stopped: %s",
                        exc,
                        exc_info=True
                    )
            finally:
                executor.shutdown()
                low_state_node.destroy_node()
            self.logger.info("Low state subscription stopped")

        self.low_state_thread = threading.Thread(
            target=low_state_service_executor,
            name="low_state_executor",
            daemon=True,
        )
        self.low_state_thread.start()

    def _vision_ball_handler(self, ball_msg):
        try:
            if not self.is_running or self.exit_event.is_set():
                return
            if getattr(ball_msg, "confidence", 0.0) > 0.0:
                self._ball_rel_xy[0] = float(ball_msg.x)
                self._ball_rel_xy[1] = float(ball_msg.y)
                self._ball_last_seen_time = time.perf_counter()
                self._ball_valid = True
        except Exception as e:
            self.logger.error(f"Error in _vision_ball_handler: {e}")

    def _loco_api_req_handler(self, rpc_req_msg):
        try:
            if not self.is_running or self.exit_event.is_set():
                return
            header = json.loads(getattr(rpc_req_msg, "header", "") or "{}")
            # Only accept explicit RotateHead RPC.
            if int(header.get("api_id", -1)) != 2004:
                return
            body = json.loads(getattr(rpc_req_msg, "body", "") or "{}")
            # moveHead RPC body contains pitch/yaw.
            if "pitch" in body and "yaw" in body:
                self._head_cmd_pitch = float(body["pitch"])
                self._head_cmd_yaw = float(body["yaw"])
                self._head_cmd_last_seen_time = time.perf_counter()
                self._head_cmd_valid = True
        except Exception:
            # Ignore unrelated RPC payloads.
            pass

    def _low_state_handler(self, low_state_msg: LowState):
        self.metrics["low_state_handler"].mark()
        try:
            if not self.is_running or self.exit_event.is_set():
                return

            # simulator tick
            self.timer.tick_timer_if_sim()

            # collect state data
            rpy = np.array(low_state_msg.imu_state.rpy, dtype=np.float32)
            gyro = np.array(low_state_msg.imu_state.gyro, dtype=np.float32)
            dof_pos = np.zeros(self.robot.num_joints, dtype=np.float32)
            dof_vel = np.zeros(self.robot.num_joints, dtype=np.float32)
            fb_torque = np.zeros(self.robot.num_joints, dtype=np.float32)

            for i, motor in enumerate(low_state_msg.motor_state_serial):
                dof_pos[i] = motor.q
                dof_vel[i] = motor.dq
                fb_torque[i] = motor.tau_est

            self._state_buf[0]["root_rpy_w"][:] = rpy
            self._state_buf[0]["root_ang_vel_b"][:] = gyro
            self._state_buf[0]["root_pos_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["root_lin_vel_w"][:] = np.zeros(
                3, dtype=np.float32
            )
            self._state_buf[0]["joint_pos"][:] = dof_pos
            self._state_buf[0]["joint_vel"][:] = dof_vel
            self._state_buf[0]["feedback_torque"][:] = fb_torque
            self.synced_state.write(self._state_buf)

            # update velocity commands to synced_command
            cmd = np.zeros((1,), dtype=self.synced_command.dtype)
            cmd[0]["vx"] = self.remoteControlService.get_vx_cmd()
            cmd[0]["vy"] = self.remoteControlService.get_vy_cmd()
            cmd[0]["vyaw"] = self.remoteControlService.get_vyaw_cmd()
            now = time.perf_counter()
            if self._ball_valid and (now - self._ball_last_seen_time) <= self._ball_timeout_s:
                cmd[0]["ball_rel_x"] = self._ball_rel_xy[0]
                cmd[0]["ball_rel_y"] = self._ball_rel_xy[1]
            else:
                cmd[0]["ball_rel_x"] = self._ball_default_rel_xy[0]
                cmd[0]["ball_rel_y"] = self._ball_default_rel_xy[1]
            if self._head_cmd_valid and (now - self._head_cmd_last_seen_time) <= self._head_cmd_timeout_s:
                cmd[0]["head_pitch_cmd"] = self._head_cmd_pitch
                cmd[0]["head_yaw_cmd"] = self._head_cmd_yaw
                cmd[0]["head_cmd_active"] = 1.0
            else:
                cmd[0]["head_pitch_cmd"] = 0.0
                cmd[0]["head_yaw_cmd"] = 0.0
                cmd[0]["head_cmd_active"] = 0.0
            self.synced_command.write(cmd)

        except Exception as e:
            self.logger.error(f"Error in _low_state_handler: {e}")
            self.running = False
            self.exit_event.set()

    def create_low_cmd_publisher(self, name):
        self.publish_node = rclpy.create_node(name)
        publisher = self.publish_node.create_publisher(
            LowCmd,
            "joint_ctrl",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST
            )
        )
        self.low_cmd_publisher = publisher

        # construct low_cmd struct
        self.low_cmd = LowCmd()  # type: ignore
        self.low_cmd.cmd_type = LowCmd.CMD_TYPE_SERIAL   # type: ignore
        motor_cmd_buf = [
            MotorCmd() for _ in range(self.robot.num_joints)
        ]  # type: ignore
        for i in range(self.robot.num_joints):
            motor_cmd_buf[i].q = 0.0
            motor_cmd_buf[i].dq = 0.0
            motor_cmd_buf[i].tau = 0.0
            motor_cmd_buf[i].kp = 0.0
            motor_cmd_buf[i].kd = 0.0
            motor_cmd_buf[i].weight = 0.0
        self.low_cmd.motor_cmd.extend(motor_cmd_buf)
        self.motor_cmd = self.low_cmd.motor_cmd

        return publisher

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        while rclpy.ok() and self.low_cmd_publisher.get_subscription_count() == 0:
            self.logger.info("Waiting for '/joint_ctrl' subscriber, retry in 0.5s")
            time.sleep(0.5)

        self.logger.info("Subscriber found, starting control loop")        

        prepare_state = self.robot.cfg.prepare_state
        init_joint_pos = self.synced_state.read()[0]['joint_pos']
        for i in range(self.robot.num_joints):
            self.motor_cmd[i].q = init_joint_pos[i]
            self.motor_cmd[i].kp = float(prepare_state.stiffness[i])
            self.motor_cmd[i].kd = float(prepare_state.damping[i])

        self.low_cmd_publisher.publish(self.low_cmd)
        time.sleep(0.1)

        # change to custom mode
        self.client.ChangeMode(RobotMode.kCustom)
        # for i in range(20):  # try multiple times to make sure mode is changed
        #     self.client.ChangeMode(RobotMode.kCustom)
        #     time.sleep(0.5)
        #     if (mode:= self.client.GetStatus().current_mode) == RobotMode.kCustom:
        #         break
        # else:
        #     self.logger.error("Failed to switch to custom mode")
        #     return False

        trans = np.linspace(init_joint_pos, prepare_state.joint_pos, num=500)
        start_time = self.timer.get_time()
        for i in range(500):
            for j in range(self.robot.num_joints):
                self.motor_cmd[j].q = trans[i][j]
            self.low_cmd_publisher.publish(self.low_cmd)
            while self.timer.get_time() < start_time + (i + 1) * 0.002:
                time.sleep(0.0002)
        self.logger.info("Custom mode started, initialized with prepare pose")
        return True

    def start_rl_gait_conditionally(self):
        """Start RL gait and spawn inference process and publisher thread."""
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while not self.exit_event.is_set():
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)

        if self.exit_event.is_set():
            return False

        # start inference process (separate process)
        self.inference_process = mp.Process(
            target=BoosterRobotPortal.inference_process_func,
            args=(
                self.cfg,
                self,
            ),
            daemon=True,
        )
        self.inference_process.start()
        self.logger.info("Inference process started")

        print(f"{self.remoteControlService.get_operation_hint()}")
        return True

    def cleanup(self) -> None:
        """Clean up resources (idempotent)."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        self.logger.info("Doing cleanup...")

        # stop threads and processes
        self.is_running = False
        self.exit_event.set()

        # wait for inference process
        if (
            self.inference_process is not None
            and self.inference_process.is_alive()
        ):
            self.logger.info("Waiting for inference process...")
            self.inference_process.join(timeout=2.0)
            if self.inference_process.is_alive():
                self.logger.warning(
                    "Inference process did not stop, terminating...")
                self.inference_process.terminate()
                self.inference_process.join(timeout=1.0)

        # close communications
        try:
            self.remoteControlService.close()
        except Exception as e:
            self.logger.error(f"Error closing remote control: {e}")

        if self.low_cmd_process is not None and self.low_cmd_process.is_alive():
            self.logger.info("Waiting for low cmd publisher process...")
            self.low_cmd_process.join(timeout=2.0)
            if self.low_cmd_process.is_alive():
                self.logger.warning(
                    "Low cmd publisher process did not stop, terminating...")
                self.low_cmd_process.terminate()
                self.low_cmd_process.join(timeout=1.0)

        try:
            thread = self.low_state_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        except Exception as e:
            self.logger.error(f"Error waiting for low state thread: {e}")

        if rclpy.ok():
            rclpy.shutdown()

        self.logger.info("Cleanup complete")

        # Print synced metrics summary to stdout
        for name, metric in self.metrics.items():
            stats = metric.compute()
            print(
                f"METRICS {name}: count={stats['count']}, "
                f"freq={stats['freq_hz']:.3f}Hz, "
                f"mean_period={stats['mean_period_s']}, "
                f"min={stats['min_period_s']}, max={stats['max_period_s']}"
            )

    def run(self):
        """Main loop: monitor inference process and diagnostics (10Hz)."""

        print("Initialization complete.")

        # start custom mode (interruptible)
        if not self.start_custom_mode_conditionally():
            print("Custom mode initialization cancelled.")
        # start RL gait (interruptible)
        elif not self.start_rl_gait_conditionally():
            print("RL gait initialization cancelled.")
        else:
            # main loop: wait for exit signal
            while self.is_running and not self.exit_event.is_set():
                # check whether the inference process is alive
                if self.inference_process is not None:
                    inference_process_alive = self.inference_process.is_alive()
                    if not inference_process_alive:
                        self.logger.error("Inference process died unexpectedly")
                        self.is_running = False
                        self.exit_event.set()
                        break
                time.sleep(0.1)

        # exit and switch to walking mode
        self.logger.info("Exiting controller, switching to walking mode...")
        self.client.ChangeMode(RobotMode.kWalking)

    def __enter__(self) -> BoosterRobotPortal:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    @staticmethod
    def inference_process_func(
        cfg: ControllerCfg,
        portal: BoosterRobotPortal,
    ) -> None:
        BoosterRobotController(cfg, portal).run()
        portal.logger.info("Inference process stopped.")


class BoosterRobotController(BaseController):
    '''Controller for Booster robots. Note that this controller runs in a
    separate process forked by BoosterRobotPortal.
    '''
    def __init__(self, cfg: ControllerCfg, portal: BoosterRobotPortal) -> None:
        super().__init__(cfg)
        self.portal = portal
        self.head_track_from_ball = False
        self.head_track_from_loco_api = True
        self.head_track_yaw_idx = 0
        self.head_track_pitch_idx = 1
        self.head_yaw_limit = [-1.0, 1.0]
        self.head_pitch_limit = [-0.35, 0.86]
        self.head_track_height_m = 0.55
        self.head_cmd_pitch = 0.0
        self.head_cmd_yaw = 0.0
        self.head_cmd_active = False
        self.head_hold_last_target = True
        self._last_head_pitch_cmd = 0.0
        self._last_head_yaw_cmd = 0.0

    def update_vel_command(self):
        cmd = self.portal.synced_command.read()[0]

        self.vel_command.lin_vel_x = cmd["vx"] * self.vel_command.vx_max
        self.vel_command.lin_vel_y = cmd["vy"] * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = cmd["vyaw"] * self.vel_command.vyaw_max

    def update_state(self) -> None:
        state = self.portal.synced_state.read()[0]

        self.robot.data.joint_pos = torch.from_numpy(
            state["joint_pos"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            state["joint_vel"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            state["feedback_torque"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            state["root_pos_w"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        rpy_t = torch.from_numpy(state["root_rpy_w"]).to(
            dtype=torch.float32).to(self.robot.data.device)
        self.robot.data.root_quat_w = lab_math.quat_from_euler_xyz(
            *rpy_t
        ).squeeze()
        self.robot.data.root_lin_vel_b = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.from_numpy(
                state["root_lin_vel_w"]).to(dtype=torch.float32).to(
                    self.robot.data.device)
        )
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            state["root_ang_vel_b"]).to(dtype=torch.float32).to(
                self.robot.data.device)
        cmd = self.portal.synced_command.read()[0]
        self.ball_rel_xy = torch.tensor(
            [cmd["ball_rel_x"], cmd["ball_rel_y"]],
            dtype=torch.float32,
            device=self.robot.data.device,
        )
        self.head_cmd_pitch = float(cmd["head_pitch_cmd"])
        self.head_cmd_yaw = float(cmd["head_yaw_cmd"])
        self.head_cmd_active = bool(cmd["head_cmd_active"] > 0.5)

    def _apply_internal_head_targets(self, dof_targets: torch.Tensor) -> None:
        yaw_cmd = None
        pitch_cmd = None
        if self.head_track_from_loco_api and self.head_cmd_active:
            pitch_cmd = self.head_cmd_pitch
            yaw_cmd = self.head_cmd_yaw
        elif self.head_track_from_ball and hasattr(self, "ball_rel_xy"):
            x = float(self.ball_rel_xy[0].item())
            y = float(self.ball_rel_xy[1].item())
            dist = max(0.05, math.hypot(x, y))
            yaw_cmd = math.atan2(y, x)
            pitch_cmd = math.atan2(self.head_track_height_m, dist)
        if yaw_cmd is None or pitch_cmd is None:
            if self.head_hold_last_target:
                yaw_cmd = self._last_head_yaw_cmd
                pitch_cmd = self._last_head_pitch_cmd
            else:
                return
        yaw_cmd = max(float(self.head_yaw_limit[0]), min(float(self.head_yaw_limit[1]), yaw_cmd))
        pitch_cmd = max(float(self.head_pitch_limit[0]), min(float(self.head_pitch_limit[1]), pitch_cmd))
        dof_targets[self.head_track_yaw_idx] = yaw_cmd
        dof_targets[self.head_track_pitch_idx] = pitch_cmd
        self._last_head_yaw_cmd = yaw_cmd
        self._last_head_pitch_cmd = pitch_cmd

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        self._apply_internal_head_targets(dof_targets)
        pass_through_idx = getattr(self, "pass_through_joint_idx", [])
        for i in range(self.robot.num_joints):
            if i in pass_through_idx:
                # Do not fight external controllers (e.g. movehead).
                self.portal.motor_cmd[i].q = float(self.robot.data.joint_pos[i].item())
                kp_val = 0.0
                kd_val = 0.0
            else:
                self.portal.motor_cmd[i].q = float(dof_targets[i].item())
                kp_val = float(self.robot.joint_stiffness[i].item())
                kd_val = float(self.robot.joint_damping[i].item())
            self.portal.motor_cmd[i].kp = kp_val
            self.portal.motor_cmd[i].kd = kd_val
        self.portal.low_cmd_publisher.publish(self.portal.low_cmd)

    def stop(self):
        super().stop()
        self.portal.exit_event.set()

    def run(self):
        self.update_state()
        if self.vel_command is not None:
            self.update_vel_command()
        self.start()
        next_inference_time = self.portal.timer.get_time()
        while self.is_running and not self.portal.exit_event.is_set():
            if self.portal.timer.get_time() < next_inference_time:
                time.sleep(0.0002)
                continue
            next_inference_time += self.cfg.policy_dt

            self.update_state()
            if self.vel_command is not None:
                self.update_vel_command()
            self.portal.metrics["policy_step"].mark()
            dof_targets = self.policy_step()
            self.ctrl_step(dof_targets)

        self.portal.exit_event.set()
