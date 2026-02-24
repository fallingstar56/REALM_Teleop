# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REALM is a simulation benchmark for evaluating generalization of robotic manipulation policies (VLA models like Pi0, Pi0-FAST, GR00T). Built on OmniGibson 1.1.1, it provides 10 manipulation tasks tested against 16 perturbation types (visual, semantic, behavioral). All execution happens inside Docker/Apptainer containers with IsaacSim.

## Directory Structure

```
REALM/
├── .docker/                          # Container definitions
│   ├── realm.Dockerfile              # Docker image (FROM stanfordvl/omnigibson:1.1.1)
│   └── realm.def                     # Apptainer definition
├── custom_assets/                    # Custom USD/simulation assets (e.g., impact_drawer)
├── examples/                         # Evaluation entry points
│   ├── 00_debug.py                   # Debug mode (zero actions)
│   ├── 00g_debug_ee_control.py       # Debug end-effector control
│   ├── 01_pi0_eval.py                # Quick single-task eval
├── packages/
│   └── openpi-client/                # WebSocket inference client (local pip package)
│       └── src/openpi_client/        # msgpack serialization, image tools
├── realm/                            # Core library
│   ├── eval.py                       # Main evaluation pipeline and entrypoint
│   ├── helpers.py                    # Transforms, object placement, image processing
│   ├── inference.py                  # Policy server client wrapper
│   ├── logging.py                    # Video recording & CSV metrics
│   ├── config/                       # All YAML configuration
│   │   ├── tasks/                    # 10 task definitions (*.yaml)
│   │   ├── robots/franka_robotiq.yaml
│   │   ├── scenes/scenes.yaml        # BEHAVIOR-1K scene spawn regions
│   │   ├── objects/categories.yaml   # Object taxonomy for perturbations
│   │   └── env/                      # Camera sensor specs & extrinsics
│   ├── environments/
│   │   ├── realm_environtment_base.py    # Base env (note: typo in filename)
│   │   ├── realm_environment_dynamic.py  # Perturbation system
│   │   └── task_progressions.py          # Stage definitions per task type
│   ├── robots/
│   │   ├── franka_robotiq.py             # Main robot class
│   │   ├── franka_robotiq_mounted.py     # Mounted variant
│   │   ├── droid_joint_controller.py     # 7-DOF joint PD control
│   │   ├── droid_gripper_controller.py   # Multi-finger gripper
│   │   ├── droid_ee_controller.py        # Cartesian EE control (IK-based)
│   │   └── robot_ik/                     # IK solvers (dm-control/dm-robotics)
│   └── misc/
│       └── modified_entity_prim.py       # Custom OmniGibson patches
└── scripts/
    ├── eval.sh                       # Comprehensive eval runner
    ├── run_docker.sh                 # Docker container launcher
    ├── run_apptainer.sh              # Apptainer container launcher
    └── cluster_evals/                # SLURM cluster evaluation scripts
```

## Running Evaluations

All commands run **inside the container** (launched via `source ./scripts/run_docker.sh`).

```bash
# Quick single-task evaluation
OMNIGIBSON_HEADLESS=1 python /app/examples/01_pi0_eval.py

# Full benchmark evaluation
OMNIGIBSON_HEADLESS=1 python /app/realm/eval.py \
    --perturbation_id 0 --task_id 0 --repeats 25 --max_steps 800 \
    --model pi0_FAST --port 8000 --experiment_name exp001

# Comprehensive multi-task evaluation script
./scripts/eval.sh -c /path/to/checkpoint -t 0 -p 0 -r 25 -m pi0_FAST
```

The model inference server (openpi) must be running separately with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to leave GPU memory for IsaacSim.

## Architecture

### Evaluation Pipeline

`realm/eval.py::evaluate()` → creates `RealmEnvironmentDynamic` + `InferenceClient` → runs rollout loop collecting metrics → saves videos/CSV/numpy artifacts to `log_dir`.

### Environment Hierarchy

- **`realm/environments/realm_environtment_base.py`** (note: typo in filename) — Base class wrapping OmniGibson. Creates scene, loads robot/objects/distractors, defines task progression stages, checks success conditions.
- **`realm/environments/realm_environment_dynamic.py`** — Extends base with 16 perturbation methods (`default()`, `v_view()`, `v_sc()`, `v_aug()`, etc.). Each perturbation modifies the environment during `reset()`.

### Configuration System

All in `realm/config/` as YAML:
- `tasks/` — 10 task configs defining objects, instructions, initial joint positions, scene locations
- `scenes/scenes.yaml` — BEHAVIOR-1K scene layouts, furniture positions, spawn regions
- `objects/categories.yaml` — Object categories for semantic perturbations
- `robots/franka_robotiq.yaml` — PD gains, control frequency (15Hz), camera resolution
- `env/` — Camera sensor specs and extrinsics

### Robot Controllers

`realm/robots/` contains modular controllers for Franka Panda + Robotiq gripper:
- `droid_joint_controller.py` — 7-DOF joint PD control (primary control mode). PD gains: Kq=[40,30,50,35,35,25,10], Kqd=[4,6,5,5,3,2,1]
- `droid_gripper_controller.py` — Multi-finger gripper (binary/smooth/independent modes). Finger range [0, 0.05] → normalized [-1, 1]
- `droid_ee_controller.py` — End-effector Cartesian control with modes: absolute_pose, pose_absolute_ori, pose_delta_ori, position_fixed_ori, position_compliant_ori, cartesian_velocity
- `robot_ik/` — IK solvers using dm-control/dm-robotics

### Inference Client

`realm/inference.py` — Websocket client connecting to remote policy server (openpi). Handles image preprocessing per model type (Pi0: 224x224 with pad, GR00T: 320x180). Debug mode returns zero actions.

### Perturbation IDs

0=Default, 1=V-AUG, 2=V-VIEW, 3=V-SC, 4=V-LIGHT, 5=S-PROP, 6=S-LANG, 7=S-MO, 8=S-AFF, 9=S-INT, 10=B-HOBJ, 11=SB-NOUN, 12=SB-VRB, 13=VB-POSE, 14=VB-MOBJ, 15=VSB-NOBJ

### Perturbation Details

| ID | Name | Method | Effect |
|----|------|--------|--------|
| 0 | Default | `default()` | No-op baseline |
| 1 | V-AUG | Applied in inference preprocessing | Gaussian blur + contrast adjustment |
| 2 | V-VIEW | `v_view()` | Random camera pose shifts (±0.2m pos, ±0.2rad rot) |
| 3 | V-SC | Dynamic distractor spawning | Adds random objects to scene |
| 4 | V-LIGHT | `v_light()` | Random light intensity (20k–750k lux) and color noise |
| 5 | S-PROP | `s_prop()` | Property-based language variation (from cached alternatives) |
| 6 | S-LANG | `s_lang()` | Synonym replacement (from cached alternatives) |
| 7 | S-MO | `s_mo()` | Spatial relationship descriptions |
| 8 | S-AFF | `s_aff()` | Affordance-based language |
| 9 | S-INT | `s_int()` | Knowledge-intensive descriptions |
| 10 | B-HOBJ | `b_hobj()` | Object mass scaling (0.25–3x), joint property changes |
| 11 | SB-NOUN | `sb_noun()` | Replace object with random distractor |
| 12 | SB-VRB | `sb_vrb()` | Switch to compatible task type (put↔pick, etc.) |
| 13 | VB-POSE | `vb_pose()` | Random object position/rotation delta |
| 14 | VB-MOBJ | `vb_mobj()` | Rescale object dimensions (0.5–1.5x per axis) |
| 15 | VSB-NOBJ | `vsb_nobj()` | Replace object with unseen category |

Perturbations follow the pattern: stop sim → modify environment → play sim → reset joints. Applied during `reset()`.

### Task IDs

0=put_green_block_in_bowl, 1=put_banana_into_box, 2=rotate_marker, 3=rotate_mug, 4=pick_spoon, 5=pick_water_bottle, 6=stack_cubes, 7=push_switch, 8=open_drawer, 9=close_drawer

### Task Progression Stages

Each task type defines an ordered sequence of stages checked sequentially (0.0–1.0):

- **put**: REACH → GRASP → LIFT_SLIGHT → MOVE_CLOSE → PLACE_INTO
- **pick**: REACH → GRASP → LIFT_LARGE
- **rotate**: REACH → GRASP → ROTATED
- **push**: REACH → TOUCH → TOGGLED_ON
- **stack**: REACH → GRASP → LIFT_SLIGHT → MOVE_CLOSE → PLACE_ONTO
- **open_drawer**: REACH → TOUCH_AND_MOVE_JOINT → OPEN_JOINT_SMALL → OPEN_JOINT_LARGE → OPEN_JOINT_FULL
- **close_drawer**: REACH → TOUCH_AND_MOVE_JOINT → CLOSE_JOINT_SMALL → CLOSE_JOINT_LARGE → CLOSE_JOINT_FULL

Success = all stages complete (progression == 1.0). Success condition methods are in `realm_environtment_base.py` (e.g., `check_reach_condition()`, `check_grasp_condition()`, `check_place_condition()`).

## Observation & Action Format

### Observations (from OmniGibson)

```python
obs = {
    'external': {
        'external_sensor0': {'rgb': array(720, 1280, 3)},  # Camera 1
        'external_sensor1': {'rgb': array(720, 1280, 3)},  # Camera 2
    },
    'franka': {
        'proprio': array([j0..j6, gripper0, gripper1]),
        'franka:gripper_link_camera:Camera:0': {'rgb': array(720, 1280, 3)},  # Wrist cam
    }
}
```

### Actions

Model outputs 8-dim: `[joint_0..joint_6, gripper_cmd]` where joints are absolute positions and gripper is 0–1. Environment converts gripper: `cmd > 0.5 → 1.0 (open), else → -1.0 (close)`.

## Metrics System

Per-episode metrics collected in `eval.py`:

- **Task**: `task_progression` (0.0–1.0), `binary_SR` (1 if complete), last completed `stage`
- **Joint dynamics**: `joint_vel_var`, `joint_acc_var`, `joint_jerk`, `joint_path_length`
- **Cartesian dynamics**: `cart_path_length`, `cart_jerk`
- **Safety**: `collisions_self`, `collisions_env`, `object_drops`

Output directory layout:
```
logs/{experiment_name}/{model}/{run_id}/
├── videos/     # MP4 rollout recordings
├── qpos/       # NumPy joint trajectories
├── actions/    # NumPy executed actions
└── reports/    # CSV with all metrics
```

## Build & Container Setup

**Docker** (recommended): `.docker/realm.Dockerfile` extends `stanfordvl/omnigibson:1.1.1`. Installs wandb, moviepy, openpi-client, dm-control, dm-robotics via micromamba + pip.

**Apptainer**: `.docker/realm.def` — alternative for HPC clusters (currently less stable).

**Setup**: `./setup.sh --docker --dataset` builds container and downloads BEHAVIOR-1K dataset (~1TB).

**Runtime**: All code runs inside container with `PYTHONPATH=/app`. Not pip-installable.

## Simulation Config

Set in `eval.py::set_sim_config()`:
- Control/render frequency: 15Hz
- Physics substep frequency: 120Hz
- `ENABLE_TRANSITION_RULES = False` (disables OG state transition bugs)
- Deterministic seeding: `seed=1234` across random, numpy, torch

## Key Conventions

- Actions are **absolute joint configurations** (7 joints + 1 gripper), not deltas
- Gripper mapping: model outputs (0,1) → environment expects (-1,1) with threshold at 0.5
- No CI/CD or formal test suite — validation is via manual evaluation runs
- `packages/openpi-client/` has minimal pytest tests for image tools and msgpack serialization

## Developer Workflows

### Adding a New Task

1. Create `realm/config/tasks/my_task.yaml` with main_objects, target_objects, distractors, cached_semantic_perturbations, instruction, task_type
2. Add stage sequence to `realm/environments/task_progressions.py`
3. Add success condition methods to `realm_environtment_base.py` if needed
4. Register task ID in `eval.py` task list

### Adding a New Perturbation

1. Add method `def my_perturbation(self):` to `RealmEnvironmentDynamic`
2. Register in `self.supported_pertrubations` dict (note: typo in attribute name)
3. Assign perturbation ID in the mapping

### Adding a New Model

1. Add preprocessing logic to `InferenceClient.extract_from_obs()` in `realm/inference.py` (image resize, normalization)
2. Update model name handling in eval scripts
