# Introduction
This repository is a fork of [REALM](https://github.com/martin-sedlacek/REALM.git) with additional features for Oculus Quest support.

We add an optional `--action_source` argument with two modes: `policy` (original) and `teleop` (new). The `teleop` mode is controlled using an Oculus VR controller.

# Usage
1. Clone the project repository:
```
git clone https://github.com/fallingstar56/REALM.git
cd REALM
```

2. Run the setup script:
```bash
# [RECOMMENDED] Docker:
./setup.sh --docker --dataset

# w/ custom dataset path:
./setup.sh --docker --dataset --data-path /path/to/dataset

# Apptainer (HPC clusters, less stable):
./setup.sh --apptainer --dataset
```

3. Open the containerized environment:
```bash
cd $REALM_ROOT
sudo ./scripts/run_docker.sh
```

4. Connect the Oculus VR:
```bash
apt update && apt install -y android-tools-adb
adb start-server
adb devices
```
> ❗ **If the device is unauthorized, you need to wear the VR handset to confirm.**

5. Installing dependencies:
```bash
pip install /app/realm/controllers/oculus_reader/requirements.txt
```

6. Run the program
```bash
python /app/examples/02_evaluate.py --experiment_name “test” --action_source “teleop” --task_id 0
```

In `teleop` mode, each successful `BUTTON B` save writes that rollout's recorded images, `task.json`, and `frames.jsonl` directly into a numbered subfolder named `<experiment_name>_<id>` under the experiment directory.

Then you can use the VR controller to teleoperate the robot.

# Tasks and Perturbations

| PERTURBATION_ID | Perturbation | Description                                                                                     | Category |
|:----------------| :--- |:------------------------------------------------------------------------------------------------| :--- |
| 0               | **Default** | Testing a skill under no specific perturbations.                                                | General |
| 1               | **V-AUG** | Randomize *blur* and *contrast*.                                                                | Visual |
| 2               | **V-VIEW** | Random shifts to external *camera pose*.                                                        | Visual |
| 3               | **V-SC** | Randomly spawn *new distractors* in the scene.                                                  | Visual |
| 4               | **V-LIGHT** | Randomize illumination *color* and *intensity*.                                                 | Visual |
| 5               | **S-PROP** | Reference objects based on their properties.                                                    | Semantic |
| 6               | **S-LANG** | Reference similar verbs and remove articles.                                                    | Semantic |
| 7               | **S-MO** | Reference spatial relationships in the scene.                                                   | Semantic |
| 8               | **S-AFF** | Reference human needs and use cases.                                                            | Semantic |
| 9               | **S-INT** | Reference facts about the world that typically require knowledge from Internet-scale text data. | Semantic |
| 10              | **B-HOBJ** | Randomize manipulated object *mass*.                                                            | Behavioral |
| 11              | **SB-NOUN** | Reference *another known object* in the scene.                                                  | Semantic + Behavioral |
| 12              | **SB-VRB** | Change the *tested skill* for another compatible one.                                           | Semantic + Behavioral |
| 13              | **VB-POSE** | Randomize manipulated *object pose*.                                                            | Visual + Behavioral |
| 14              | **VB-MOBJ** | Randomize object *size* and *shape*.                                                            | Visual + Behavioral |
| 15              | **VSB-NOBJ** | Sample a *new unseen manipulated object*.                                                       | Visual + Semantic + Behavioral |

| TASK_ID | Task |
|:--------| :--- |
| 0       | put_green_block_in_bowl |
| 1       | put_banana_into_box |
| 2       | rotate_marker |
| 3       | rotate_mug |
| 4       | pick_spoon |
| 5       | pick_water_bottle |
| 6       | stack_cubes |
| 7       | push_switch |
| 8       | open_drawer |
| 9       | close_drawer |

