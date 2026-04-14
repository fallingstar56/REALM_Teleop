from queue import Queue
import datetime
import time
import os
import random
import csv
import json
import shutil
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

import omnigibson as og
from omnigibson.macros import gm

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient, extract_from_obs
from realm.realm_logging import VideoRecorder, save_results
from realm.controllers.oculus_controller import VRPolicy


SUPPORTED_TASKS = [
    "put_green_block_into_bowl", #0
    "put_banana_into_box", #1
    "rotate_marker", #2
    "rotate_mug", #3
    "pick_spoon", #4
    "pick_water_bottle", #5
    "stack_cubes", #6
    "push_switch", #7
    "open_drawer", #8
    "close_drawer", #9
]

SUPPORTED_PERTURBATIONS = [
    'Default', #0
    'V-AUG', # 1
    'V-VIEW',  # 2
    'V-SC', # 1
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'SB-NOUN', # 11
    'SB-VRB', # 12
    'VB-POSE',  # 13
    'VB-MOBJ',  # 14
    'VSB-NOBJ' # 15
]

SUPPORTED_ACTION_SOURCES = ["policy", "teleop"]
# policy: get actions from a model inference server
# teleop: get actions from  OCulus VR controller teleoperation (currently only supports DROID EE controller config with cartesian velocity control)

RECORD_INTERVAL_SEC = 0.05


def _get_next_saved_rollout_id(parent_dir, experiment_name):
    if experiment_name is None:
        return 0

    max_saved_rollout_id = -1
    prefix = f"{experiment_name}_"

    if not os.path.isdir(parent_dir):
        return 0

    for entry_name in os.listdir(parent_dir):
        entry_path = os.path.join(parent_dir, entry_name)
        if not os.path.isdir(entry_path) or not entry_name.startswith(prefix):
            continue

        saved_rollout_id = entry_name[len(prefix):]
        if saved_rollout_id.isdigit():
            max_saved_rollout_id = max(max_saved_rollout_id, int(saved_rollout_id))

    return max_saved_rollout_id + 1


def _get_rollout_storage_dir(log_dir, action_source, experiment_name=None, experiment_root_dir=None, saved_rollout_id=None):
    if action_source != "teleop" or experiment_name is None or saved_rollout_id is None:
        return log_dir

    parent_dir = experiment_root_dir if experiment_root_dir is not None else log_dir
    return os.path.join(parent_dir, f"{experiment_name}_{saved_rollout_id}")


def _ensure_uint8_hwc(img):
    img = np.asarray(img)

    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = img.transpose(1, 2, 0)

    if img.dtype != np.uint8:
        img_float = img.astype(np.float32)
        if img_float.size > 0 and img_float.max() <= 1.5:
            img_float = img_float * 255.0
        img = np.clip(img_float, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img


def _save_jpg(path, img_u8, quality=95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8).save(path, format="JPEG", quality=quality, subsampling=0)


def _record_rollout_sample(
    frames_file,
    frame_index,
    img_dir,
    img_sec_dir,
    wrist_dir,
    base_im,
    base_im_second,
    wrist_im,
    robot_state,
    gripper_state,
    action,
):
    base_u8 = _ensure_uint8_hwc(base_im)
    base_sec_u8 = _ensure_uint8_hwc(base_im_second if base_im_second is not None else np.zeros_like(base_u8))
    wrist_u8 = _ensure_uint8_hwc(wrist_im)

    img_name = f"{frame_index:06d}.jpg"

    _save_jpg(os.path.join(img_dir, img_name), base_u8)
    _save_jpg(os.path.join(img_sec_dir, img_name), base_sec_u8)
    _save_jpg(os.path.join(wrist_dir, img_name), wrist_u8)

    frame_obj = {
        "index": frame_index,
        "image": f"image/{img_name}",
        "image_sec": f"image_sec/{img_name}",
        "wrist_image": f"wrist_image/{img_name}",
        "robot_state": np.asarray(robot_state, dtype=np.float32).tolist(),
        "gripper_state": np.atleast_1d(np.asarray(gripper_state, dtype=np.float32)).tolist(),
        "action": np.asarray(action, dtype=np.float32).tolist(),
    }
    frames_file.write(json.dumps(frame_obj, ensure_ascii=False) + "\n")


def _cleanup_rollout_recording(
    video_recorder=None,
    frames_file=None,
    info_dir=None,
    discard_info=False,
    rollout_dir=None,
    discard_rollout_dir=False,
):
    if frames_file is not None and not frames_file.closed:
        frames_file.close()

    if video_recorder is not None:
        video_recorder.cleanup()

    if discard_info and info_dir is not None and os.path.isdir(info_dir):
        shutil.rmtree(info_dir, ignore_errors=True)

    if discard_rollout_dir and rollout_dir is not None and os.path.isdir(rollout_dir):
        shutil.rmtree(rollout_dir, ignore_errors=True)

def set_sim_config(rendering_mode=None, robot="DROID"):
    if robot == "WidowX": # TODO: just read this from the yamls...
        gm.DEFAULT_SIM_STEP_FREQ = 5
        gm.DEFAULT_RENDERING_FREQ = 5
    elif "UR5" in robot:
        gm.DEFAULT_SIM_STEP_FREQ = 30
        gm.DEFAULT_RENDERING_FREQ = 30
    else:
        gm.DEFAULT_SIM_STEP_FREQ = 15
        gm.DEFAULT_RENDERING_FREQ = 30  

    gm.DEFAULT_PHYSICS_FREQ = 120
    gm.ENABLE_TRANSITION_RULES = False # this needs to be off to avoid bug with sludge state during collision: https://github.com/StanfordVL/BEHAVIOR-1K/issues/1201
    gm.ENABLE_OBJECT_STATES = True # this needs to be on because push_switch task usees the ToggledOn state
    gm.RENDER_VIEWER_CAMERA=False
    gm.ENABLE_HQ_RENDERING = False if rendering_mode == "r" else True

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(
        task_id=0,
        perturbation_id=0,
        repeats=1,
        max_steps=500,
        horizon=8,
        model_type="pi0_FAST",
        port=8000,
        host="127.0.0.1",
        log_dir="/app/logs",
        resume=False,
        multi_view=False,
        no_record=False,
        no_render=False,
        rendering_mode=None,
        task_cfg_path=None,
        robot="DROID",
        action_source="policy",
        experiment_name=None,
        experiment_root_dir=None,
):
    start = time.perf_counter()
    og.log.info(f"DEBUG: Begin eval: {time.perf_counter() - start:.4f}s")
    if rendering_mode is None:
        rendering_mode = "rt"
    set_sim_config(rendering_mode=rendering_mode, robot=robot)

    if action_source not in SUPPORTED_ACTION_SOURCES:
        raise ValueError(
            f"Unsupported action_source={action_source!r}. Expected one of {SUPPORTED_ACTION_SOURCES}."
        )

    # -------------------- Create the environment + client --------------------
    if task_cfg_path is None:
        task = SUPPORTED_TASKS[task_id]
        task_cfg_path = f"REALM_DROID10/{task}/default.yaml"
    else:
        task = task_cfg_path.split("/")[-2]
        config_name = task_cfg_path.split("/")[-1].replace(".yaml", "").replace(".cfg", "")
        if config_name != "default":
            task = f"{task}_{config_name}"

    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    os.makedirs(log_dir, exist_ok=True)
    # if not no_record:
    #     os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)
    #     os.makedirs(os.path.join(log_dir, "information"), exist_ok=True)

    next_saved_rollout_id = None
    if action_source == "teleop" and experiment_name is not None:
        teleop_rollout_parent_dir = experiment_root_dir if experiment_root_dir is not None else log_dir
        os.makedirs(teleop_rollout_parent_dir, exist_ok=True)
        next_saved_rollout_id = _get_next_saved_rollout_id(teleop_rollout_parent_dir, experiment_name)

    client = None
    if action_source == "policy":
        model_type = model_type # TODO: infer type from model name, rn this will just default to a pi model inference inside the client
        client = InferenceClient(model_type, host=host, port=port)
        og.log.info(f"DEBUG: Client connected: {time.perf_counter() - start:.4f}s")
    
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=perturbations,
        multi_view=multi_view,
        no_rendering=no_render,
        rendering_mode=rendering_mode,
        robot=robot
    )
    
    if action_source == "teleop" and not env.ee_control:
        raise ValueError(
            "VR teleop requires ee_control=true and DroidEndEffectorController. "
            "Please use DROID EE controller config."
        )
    # Now we use DROID, and the controller name is DroidEndEffectorController, the mode is cartesian_velocity
    og.log.info(f"DEBUG: Env created: {time.perf_counter() - start:.4f}s")

    results = []
    start_repeat = 0
    results_filename = None
    effective_repeats = repeats

    # if resume:
    #     potential_csv = os.path.join(log_dir, "reports", f"{task}_{perturbations[0]}.csv")
    #     if os.path.exists(potential_csv):
    #         results_filename = potential_csv
    #         with open(results_filename, 'r') as f:
    #             reader = csv.DictReader(f)
    #             existing_results = list(reader)
    #         results = existing_results
    #         start_repeat = len(results)
    #         og.log.info(f"Resuming run from repeat {start_repeat}. Using file: {results_filename}")
    #     else:
    #         og.log.info(f"Resume requested but no report found. Starting fresh.")

    for run_id in range(effective_repeats):
        # ------------------------ pre-configure each run --------------------------------
        # seed = 1234 + run_id
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        if run_id < start_repeat:
            continue

        controller = None
        last_reset_pressed = False
        last_save_pressed = False
        if action_source == "teleop":
            controller = VRPolicy(robot_base_yaw=env.robot_rot_rad[2])

        while True:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            rollout_name = f"{timestamp}_{model_type}_rollout_{task}_{perturbations[0]}_{run_id}"
            rollout_storage_dir = _get_rollout_storage_dir(
                log_dir,
                action_source=action_source,
                experiment_name=experiment_name,
                experiment_root_dir=experiment_root_dir,
                saved_rollout_id=next_saved_rollout_id,
            )
            # video_recorder = VideoRecorder(rollout_storage_dir, timestamp, run_id, task, perturbations[0]) if not no_record else None
            video_recorder = None

            qpos = []
            actions = []
            action_buffer = Queue()
            frames_f = None
            info_dir = None
            img_dir = None
            img_sec_dir = None
            wrist_dir = None
            last_record_time = -1e9
            frame_index = 0

            # -------------------- Rollout loop --------------------
            obs, _ = env.reset()
            if not no_record:
                info_dir = rollout_storage_dir
                img_dir = os.path.join(info_dir, "image")
                img_sec_dir = os.path.join(info_dir, "image_sec")
                wrist_dir = os.path.join(info_dir, "wrist_image")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(img_sec_dir, exist_ok=True)
                os.makedirs(wrist_dir, exist_ok=True)

                with open(os.path.join(info_dir, "task.json"), "w", encoding="utf-8") as task_file:
                    json.dump({"task": str(env.instruction)}, task_file, ensure_ascii=False)

                frames_f = open(os.path.join(info_dir, "frames.jsonl"), "w", encoding="utf-8")

            obs, rew, terminated, truncated, info = env.warmup(obs)

            t = 0
            task_progression = 0.0
            task_progression_timestamps = []
            terminal_steps = 15
            enforce_terminal_step_limit = action_source != "teleop"
            enforce_max_steps_limit = action_source != "teleop"

            ee_poses = []
            collisions_self = 0
            collisions_env = 0
            is_self_col_active = False
            is_env_col_active = False
            drops = 0
            was_grasping = False
            restart_requested = False
            save_requested = False

            while (not enforce_max_steps_limit or t < max_steps) and (not enforce_terminal_step_limit or terminal_steps > 0):
                # Extract the relevant information from the observation for the model
                base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs, robot_name=env.robot.name)
            
                # Metrics collection
                ee_pos, ee_rot = env.get_ee_pose()
                ee_poses.append(ee_pos)
                _ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
                _ee_rot = ee_rot.cpu().numpy() if hasattr(ee_rot, 'cpu') else np.array(ee_rot)
                _ee_euler = Rot.from_quat(_ee_rot).as_euler('xyz')
                _ee_pos_world = np.concatenate([_ee_pos, _ee_euler])
                cartesian_position = env._world2robot(_ee_pos_world).astype(np.float32)

                _, fp_rot = env.get_first_person_pose()
                _fp_rot = fp_rot.cpu().numpy() if hasattr(fp_rot, 'cpu') else np.array(fp_rot)
                first_person_quat = (
                    Rot.from_euler('z', -float(env.robot_rot_rad[2])) * Rot.from_quat(_fp_rot)
                ).as_quat().astype(np.float32)

                # Check for collisions
                is_self_col, is_env_col = env.check_collisions()
                if is_self_col and not is_self_col_active:
                    collisions_self += 1
                is_self_col_active = is_self_col

                if is_env_col and not is_env_col_active:
                    collisions_env += 1
                is_env_col_active = is_env_col

                is_grasping = env.check_grasp_condition(obs)
                if was_grasping and not is_grasping:
                    is_placed = False
                    if hasattr(env, "task_type") and env.task_type in ["put", "stack"] and len(env.target_objects) > 0:
                        mo = env.main_objects[0]
                        target = env.target_objects[0]
                        inside = mo.states[og.object_states.Inside].get_value(target)
                        on_top = mo.states[og.object_states.OnTop].get_value(target)
                        if inside or on_top:
                            is_placed = True

                    if not is_placed:
                        drops += 1
                was_grasping = is_grasping

                if action_source == "policy":
                    if action_buffer.empty():
                        pred_action_chunk = client.infer(
                            env.instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
                            use_base_im_second=(env.task_type == "open_close_drawer" if hasattr(env, "task_type") else False),
                            ee_control=env.ee_control,
                            cartesian_position=cartesian_position
                        )

                        if len(pred_action_chunk.shape) == 2:
                            for action in pred_action_chunk[:horizon]:
                                action = np.squeeze(action)
                                action_buffer.put(action)
                        elif len(pred_action_chunk.shape) < 2:
                            action_buffer.put(pred_action_chunk)
                        else:
                            assert len(pred_action_chunk.shape) <= 2, f"Unsupported number of dimensions in action chunk with shape: {pred_action_chunk.shape}. The chunk is expected to be 2D."

                    action = action_buffer.get()
                else:
                    state_dict = {
                        "cartesian_position": cartesian_position,
                        "gripper_position": gripper_state,
                        "first_person_quat": first_person_quat,
                    }

                    controller_info = controller.get_info()
                    reset_pressed = controller_info["success"]
                    reset_triggered = reset_pressed and not last_reset_pressed
                    last_reset_pressed = reset_pressed

                    save_pressed = controller_info["failure"]
                    save_triggered = save_pressed and not last_save_pressed
                    last_save_pressed = save_pressed

                    if reset_triggered:
                        og.log.info("Teleop reset requested from BUTTON A. Restarting current rollout.")
                        controller.reset_state()
                        restart_requested = True
                        break

                    if save_triggered:
                        og.log.info("Teleop save requested from BUTTON B. Finalizing current rollout and resetting.")
                        save_requested = True
                        break

                    has_pose = controller._state["poses"] != {}

                    if ((not has_pose) or (not controller_info["controller_on"]) or (not controller_info["movement_enabled"])):
                        action = controller.get_idle_action()
                    else:
                        action = controller._calculate_action(state_dict, False).astype(np.float32)
                        action = np.clip(action, -1.0, 1.0)

                if action_source == "policy":
                    executed_action = np.asarray(action, dtype=np.float32).copy()
                    if model_type in ["debug", "openpi", "GR00T", "GR00T_N16", "dreamzero"]: # TODO: use a model config
                        executed_action[-1] = 1 if action[-1] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
                    elif model_type == "molmoact":
                        executed_action[-1] = 1 if action[-1] < 0.5 else -1  # Prediction: (0,1) -> Target: (1,-1)
                    else:
                        raise NotImplementedError()
                else:
                    executed_action = np.asarray(action, dtype=np.float32).copy()

                if not no_record:
                    # video_recorder.add_frame(base_im, wrist_im, base_im_second)
                    current_time = time.perf_counter()
                    if (current_time - last_record_time) >= RECORD_INTERVAL_SEC:
                        last_record_time = current_time
                        _record_rollout_sample(
                            frames_f,
                            frame_index,
                            img_dir,
                            img_sec_dir,
                            wrist_dir,
                            base_im,
                            base_im_second,
                            wrist_im,
                            robot_state,
                            gripper_state,
                            executed_action,
                        )
                        frame_index += 1

                qpos.append(np.concatenate((robot_state, np.atleast_1d(np.asarray(gripper_state)))))
                actions.append(executed_action)

                obs, curr_task_progression, terminated, truncated, info = env.step(executed_action)

                if curr_task_progression > task_progression:
                    task_progression = curr_task_progression
                    task_progression_timestamps.append(t)
                if task_progression >= 1.0:
                    terminal_steps -= 1
                t += 1

            if restart_requested:
                if client is not None:
                    client.reset()
                _cleanup_rollout_recording(
                    video_recorder=video_recorder,
                    frames_file=frames_f,
                    info_dir=info_dir,
                    discard_info=True,
                    rollout_dir=rollout_storage_dir,
                    discard_rollout_dir=rollout_storage_dir != log_dir,
                )
                continue

            if save_requested and len(qpos) == 0:
                og.log.warning("Teleop save requested before any samples were collected. Resetting without writing recorded files.")
                _cleanup_rollout_recording(
                    video_recorder=video_recorder,
                    frames_file=frames_f,
                    info_dir=info_dir,
                    discard_info=True,
                    rollout_dir=rollout_storage_dir,
                    discard_rollout_dir=rollout_storage_dir != log_dir,
                )
                if client is not None:
                    client.reset()
                continue

            og.log.info(f"DEBUG: Run finished: {time.perf_counter() - start:.4f}s")
            # ------------------------------------------------------------------------------

            # Metrics calculation
            dt = 1.0 / 15.0  # Control freq is 15Hz by default

            qpos_arr = np.stack(qpos)  # (N, 8)
            qpos_joints = qpos_arr[:, :7]

            # Joint space metrics
            if len(qpos_joints) > 4:
                joint_vel = np.diff(qpos_joints, axis=0) / dt
                joint_acc = np.diff(joint_vel, axis=0) / dt
                joint_jerk = np.diff(joint_acc, axis=0) / dt

                joint_vel_var = np.mean(np.var(joint_vel, axis=0) * len(joint_vel))
                joint_acc_var = np.mean(np.var(joint_acc, axis=0) * len(joint_acc))
                joint_jerk_metric = np.mean(np.linalg.norm(joint_jerk, axis=1))
                joint_path_length = np.sum(np.linalg.norm(np.diff(qpos_joints, axis=0), axis=1))
            else:
                joint_vel_var = 0.0
                joint_acc_var = 0.0
                joint_jerk_metric = 0.0
                joint_path_length = 0.0

            # Cartesian space metrics
            ee_pos_arr = np.stack(ee_poses)
            if len(ee_pos_arr) > 4:
                cart_vel = np.diff(ee_pos_arr, axis=0) / dt
                cart_acc = np.diff(cart_vel, axis=0) / dt
                cart_jerk = np.diff(cart_acc, axis=0) / dt

                cart_jerk_metric = np.mean(np.linalg.norm(cart_jerk, axis=1))
                cart_path_length = np.sum(np.linalg.norm(np.diff(ee_pos_arr, axis=0), axis=1))
            else:
                cart_path_length = 0.0
                cart_jerk_metric = 0.0

            stage_to_log = "SUCCESS"
            if env.task_progression is not None:
                for stage, is_completed in env.task_progression.items():
                    if not is_completed:
                        stage_to_log = stage
                        break
            else:
                stage_to_log = "N/A"

            if task_progression == 1.0 and hasattr(env, "task_type") and env.task_type in ["put", "stack"]:
                drops = max(0, drops - 1)

            result_entry = {
                "run_id": run_id,
                "task": task,
                "perturbation": perturbations[0],
                "instruction": env.instruction,
                "model": model_type,
                "action_source": action_source,
                "real2sim": "Simulated",
                "env": "REALM",
                "task_progression": task_progression,
                "task_progression_timestamps": task_progression_timestamps,
                "stage": stage_to_log,
                "binary_SR": 1.0 if task_progression == 1.0 else 0.0,
                "joint_vel_var": joint_vel_var,
                "joint_acc_var": joint_acc_var,
                "joint_jerk": joint_jerk_metric,
                "joint_path_length": joint_path_length,
                "cart_path_length": cart_path_length,
                "cart_jerk": cart_jerk_metric,
                "collisions_self": collisions_self,
                "collisions_env": collisions_env,
                "object_drops": drops
            }

            result_entry["qpos"] = np.stack(qpos).tolist()
            result_entry["actions"] = np.stack(actions).tolist()
        
            results.append(result_entry)

            # if not no_record:
            #     video_recorder.save_video(os.path.join(rollout_storage_dir, "videos", rollout_name))

            _cleanup_rollout_recording(video_recorder=video_recorder, frames_file=frames_f)

            if action_source == "teleop" and rollout_storage_dir != log_dir:
                # save_results([result_entry], os.path.join(rollout_storage_dir, "reports"), task, perturbations[0])
                og.log.info(f"Saved teleop rollout artifacts to {rollout_storage_dir}")
                next_saved_rollout_id += 1

            if client is not None:
                client.reset()

            # results_filename = save_results(results, log_dir + "/reports", task, perturbations[0], filename=results_filename)
            break

    # ------------------------------------------------------------------------------
    # save_results(results, log_dir+"/reports", task, perturbations[0])
    og.log.info("Done!")
    og.log.info(f"DEBUG: Done: {time.perf_counter() - start:.4f}s")

