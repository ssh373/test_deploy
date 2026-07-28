# K1 GoTo deployment

Copy the recurrent TorchScript produced by the training `play.py` command to
`models/k1_goto_jit.pt`, then run from the `deploy` directory:

```powershell
python scripts/deploy.py --task k1_goto --mujoco
```

The terminal's three inputs are **a new body-relative pose goal**, not velocity:

```text
dx_metres dy_metres delta_yaw_radians
```

For example, `1.0 0.0 0.0` latches a point one metre ahead; `0 0 1.57` requests
a left turn. Re-entering exactly the same triple does not create a new goal;
enter `0 0 0` first when the same relative displacement should be commanded
again.

The deployment contract matches training: 200 Hz physics, 50 Hz policy, 46
actor observations, 12 leg actions, training joint order, per-joint action
scale, and nominal leg PD gains (hip/knee 100/2, ankle 50/1). MuJoCo provides
world position and yaw directly. The current real-robot portal does not provide
world translation (`root_pos_w` is zero), so this task subscribes to the real
robot's `booster_interface/msg/Odometer` topic `/odometer_state` and uses its
`x`, `y`, and `theta` for goal tracking. IMU orientation remains the source for
projected gravity and fall detection. Real deployment refuses to start without
odometry and stops if it is stale for more than 0.5 seconds.

On the real robot, relative goals are one-shot ROS events (the continuous
keyboard velocity channel is ignored by GoTo):

```bash
ros2 topic pub --once /goto/relative_goal geometry_msgs/msg/Pose2D \
  "{x: 0.1, y: 0.0, theta: 0.0}"
```

Every received message is latched from the odometry pose at callback time, so
publishing the same Pose2D again creates another relative movement. MuJoCo keeps
using its existing terminal input and does not create these ROS subscriptions.

The exported policy controls only the 12 leg joints. The GoTo training URDF
declares all 10 head/arm joints fixed at zero. MuJoCo's K1 XML models them as
movable hinges, so deployment holds that zero pose with the robot's prepare PD
gains rather than using the folded-arm pose from other walking tasks.
