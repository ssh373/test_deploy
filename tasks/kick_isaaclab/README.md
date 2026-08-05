# K1 Isaac Lab kick policy in MuJoCo

This task deploys the 49-dimensional actor trained with
`Booster-K1-Kick_001-v0`. It is separate from the older
`tasks/locomotion/kick_k1.py`, whose observation contract is incompatible.

## Model

Export the RSL-RL policy from Isaac Lab, then copy the TorchScript file to:

```text
tasks/kick_isaaclab/models/k1_kick_jit.pt
```

The actor contract is:

```text
gravity(3), base angular velocity(3), ball relative XY(2),
target relative XY * 0.25(2), visible/age/confidence(3),
joint position error(12), joint velocity * 0.1(12), last action(12)
```

Isaac Lab applies the target term's `clip=(-2, 2)` before its `scale=0.25`;
deployment preserves that exact order, so a 4 m forward target enters the
actor as `0.5`, matching the exported observation normalizer.

Ball velocity is not included in the actor observation.
Actor outputs are currently not clipped, matching the existing unclipped kick
training runs. Add the same clip to training and deployment together later.
MuJoCo holds the training default pose for 0.4 seconds after reset before
enabling policy actions, giving contacts and the base time to settle.

## Run

From the `deploy` directory:

```bash
python scripts/deploy.py --task k1_kick_isaaclab --mujoco
```

To separate an MJCF/default-pose problem from a policy mapping problem, run:

```bash
python scripts/deploy.py --task k1_kick_isaaclab_hold --mujoco
```

This uses the same model and PD controller but holds the default pose instead
of applying policy actions. The normal task prints its first ten action vectors.

To test an existing policy that was trained without action clipping while
protecting MuJoCo from its out-of-range targets, run:

```bash
python scripts/deploy.py --task k1_kick_isaaclab_clipped --mujoco
```

The default ball is 0.275 m in front of the robot and the target is 4 m ahead.
For a directional test, change `target_angle_deg` in `K1IsaacLabKickPolicyCfg`
to a value between -60 and +60 degrees.

The task owns its MJCF at `assets/K1_22dof_kick_ball.xml`. Its compiler line
expects K1 meshes at `/home/user/booster_assets/robots/K1/meshes`; update only
that `meshdir` if the Booster assets live elsewhere on the deployment machine.
