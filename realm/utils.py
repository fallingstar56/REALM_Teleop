import os
import glob
import h5py
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic


def set_flat_physics_params(env: RealmEnvironmentDynamic, flat_params: np.ndarray):
    """
    Applies flattened friction and armature parameters to the robot's arm joints.
    flat_params: [friction_0, ..., friction_n, armature_0, ..., armature_n]
    """
    # Use the robot's joint names to find joints in the stage
    # This is more robust than hardcoded paths
    joint_names = env.robot.arm_joint_names[env.robot.default_arm]
    dof = len(joint_names)
    
    for idx, j_name in enumerate(joint_names):
        if j_name in env.robot.joints:
            joint_prim = env.robot.joints[j_name].prim
            # Apply friction
            joint_prim.GetAttribute("physxJoint:jointFriction").Set(float(flat_params[idx]))
            # Apply armature
            joint_prim.GetAttribute("physxJoint:armature").Set(float(flat_params[idx + dof]))
        else:
            print(f"Warning: Joint {j_name} not found in robot joints.")


def get_joint_data(f, root_key, arm_priority=['arm_left_position_align', 'arm_right_position_align', 'arm_position_align']):
    """Helper to find joint position data in the HDF5 file."""
    if root_key not in f:
        return None
    
    group = f[root_key]
    # Try prioritized keys
    for arm_key in arm_priority:
        if arm_key in group:
            if 'data' in group[arm_key]:
                return group[arm_key]['data'][:]
    
    # Fallback: search for any key containing 'position' and having 'data'
    for k in group.keys():
        if 'position' in k and 'data' in group[k]:
            return group[k]['data'][:]
            
    return None


def replay_traj(env: RealmEnvironmentDynamic, trajectory_actions, trajectory_gt_qpos, trajectory_gt_ee=None, max_steps=1000, dof=7):
    max_steps = min(len(trajectory_actions), max_steps)

    qpos = []
    video = [] if not env.no_rendering else None
    ee_pos_list = []
    ee_rot_list = []

    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)

    # Simple warmup: go to initial GT position
    for _ in range(150):
        action = np.concatenate((trajectory_gt_qpos[0, :dof], np.atleast_1d(np.zeros(1))))
        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    for t in range(max_steps):
        base_im = None if env.no_rendering else obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        if base_im is not None and video is not None:
            video.append(base_im)

        robot_state = obs[env.robot.name]['proprio'].cpu().numpy()
        qpos.append(robot_state[:dof])

        ee_pos, ee_rot = env.get_ee_pose()
        ee_pos_list.append(ee_pos)
        ee_rot_list.append(ee_rot)

        action = np.concatenate((trajectory_actions[t, :dof], np.atleast_1d(np.zeros(1))))
        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    # Save debug replay video:
    if video is not None:
        video = np.stack(video)
        save_filename = f"/app/logs/debug_ur5_replay"
        ImageSequenceClip(list(video), fps=15).write_videofile(save_filename + ".mp4", codec="libx264")

    # Stack trajectories
    qpos_joints = np.stack(qpos)
    ee_pos_arr = np.stack(ee_pos_list)
    ee_rot_arr = np.stack(ee_rot_list)

    # Calculate errors
    # Note: ensure GT matches the length of replayed steps
    qpos_err = qpos_joints - trajectory_gt_qpos[:max_steps, :dof]
    
    ee_pos_err = None
    if trajectory_gt_ee is not None:
        ee_pos_err = ee_pos_arr - trajectory_gt_ee[:max_steps, :3]

    return {
        "qpos_err": qpos_err,
        "ee_pos_err": ee_pos_err,
        "qpos_joints": qpos_joints,
        "ee_pos": ee_pos_arr,
        "ee_rot": ee_rot_arr,
        "trajectory_gt_qpos": trajectory_actions,
        "trajectory_gt_ee": trajectory_gt_ee,
    }


def cost_function(env: RealmEnvironmentDynamic, traj_path: str, max_eps: int = 5, dof: int = 7, seed: int = 42):
    """
    Calculates the cumulative error cost over multiple trajectories.
    Supports both Droid-style .npy directories and RoboMIND-style HDF5 files.
    """
    # Determine if we are dealing with HDF5 or NPY
    hdf5_files = glob.glob(os.path.join(traj_path, "**/trajectory.hdf5"), recursive=True)
    
    if hdf5_files:
        # HDF5 Mode (UR5 / RoboMIND)
        random.seed(seed)
        random.shuffle(hdf5_files)
        ep_paths = hdf5_files[:max_eps]
        ep_mode = 'hdf5'
    else:
        # NPY Mode (Droid / Franka)
        ep_names = [d for d in os.listdir(traj_path) if os.path.isdir(os.path.join(traj_path, d))]
        random.seed(seed)
        random.shuffle(ep_names)
        ep_paths = ep_names[:max_eps]
        ep_mode = 'npy'

    cost = 0.0
    evaluated_eps = 0
    for ep in ep_paths:
        try:
            if ep_mode == 'hdf5':
                with h5py.File(ep, 'r') as f:
                    traj_qpos_actions = get_joint_data(f, 'puppet')
                    if traj_qpos_actions is None: traj_qpos_actions = get_joint_data(f, 'master')
                    
                    traj_qpos_gt = get_joint_data(f, 'master')
                    if traj_qpos_gt is None: traj_qpos_gt = traj_qpos_actions
                    
                    traj_ee_gt = None # Optional
            else:
                full_ep_path = os.path.join(traj_path, ep)
                traj_qpos_actions = np.load(os.path.join(full_ep_path, "action_joint_position.npy"))
                traj_qpos_gt = np.load(os.path.join(full_ep_path, "observation_state_joint_position.npy"))
                traj_ee_gt = np.load(os.path.join(full_ep_path, "observation_state_cartesian_position.npy"))

            if traj_qpos_actions is None or traj_qpos_gt is None:
                continue

            res_dict = replay_traj(env, traj_qpos_actions, traj_qpos_gt, traj_ee_gt, dof=dof)

            # Average MSE cost per step
            ep_cost = 10 * np.mean(np.square(res_dict["qpos_err"])) # upweight qpos error over EE error due to magnitude
            if res_dict["ee_pos_err"] is not None:
                ep_cost += np.mean(np.square(res_dict["ee_pos_err"]))
            cost += ep_cost
            print(ep, ep_cost)
            evaluated_eps += 1
        except Exception as e:
            print(f"Error evaluating episode {ep}: {e}")
            continue

    return cost / max(1, evaluated_eps)

def plot_err(res_dict, ep_name, log_dir, plot_title=None):
    plot_title = ep_name if plot_title is None else plot_title
    qpos_err = res_dict["qpos_err"]
    ee_pos_err = res_dict["ee_pos_err"]
    dof = qpos_err.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot joint errors
    axes[0].plot(qpos_err)
    axes[0].set_title(f"Joint Position Error: {plot_title}")
    axes[0].set_ylabel("Error (rad)")
    axes[0].set_xlabel("Time steps")
    axes[0].legend([f"Joint {i}" for i in range(dof)], loc='upper right')
    axes[0].grid(True)
    
    # Auto-scale if error is small
    if np.max(np.abs(qpos_err)) < 0.2:
        axes[0].set_ylim(-0.1, 0.1)

    # Plot EE xyz errors
    if ee_pos_err is not None:
        axes[1].plot(ee_pos_err)
        axes[1].set_title(f"EE XYZ Errors: {plot_title}")
        axes[1].set_ylabel("Error (m)")
        axes[1].set_xlabel("Time steps")
        axes[1].legend(['X', 'Y', 'Z'], loc='upper right')
        axes[1].grid(True)
        if np.max(np.abs(ee_pos_err)) < 0.1:
            axes[1].set_ylim(-0.05, 0.05)
    else:
        axes[1].text(0.5, 0.5, "EE Ground Truth Not Available", ha='center', va='center')

    plt.tight_layout()

    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{ep_name}.png")
    plt.savefig(plot_path)
    plt.close(fig)
