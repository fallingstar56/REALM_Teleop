import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import omnigibson as og
import omnigibson.lazy as lazy

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.inference import extract_from_obs


def set_flat_physics_params(env: RealmEnvironmentDynamic, flat_params: np.ndarray):
    joint_names = env.robot.arm_joint_names
    for idx in range(7):
        prim_path = f"{env.robot.prim_path}/panda_link{idx}/{joint_names['0'][idx]}"
        joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
        assert joint_prim.IsValid()
        joint_prim.GetAttribute("physxJoint:jointFriction").Set(flat_params[idx])
        joint_prim.GetAttribute("physxJoint:armature").Set(flat_params[idx + 7])


def replay_traj(env: RealmEnvironmentDynamic, trajectory_actions, trajectory_gt_qpos, trajectory_gt_ee):
    max_steps = min(len(trajectory_actions), 1000)

    qpos = []
    ee_pos_list = []

    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)

    for _ in range(150):
        action = trajectory_gt_qpos[0]
        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    for t in range(max_steps):
        base_im, base_im_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs)

        ee_pos, ee_rot = env.get_ee_pose()
        ee_pos_list.append(ee_pos)

        qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))

        action = trajectory_actions[t]

        obs, curr_task_progression, terminated, truncated, info = env.step(action)

    # Stack final achieve trajectories:
    qpos_arr = np.stack(qpos)  # (N, 8)
    qpos_joints = qpos_arr[:, :7]
    ee_pos_arr = np.stack(ee_pos_list)


    qpos_err= qpos_joints[:, :7] - trajectory_gt_qpos[:, :7]
    ee_pos_err = ee_pos_arr[:, :] - trajectory_gt_ee[:, :3]

    return {
        "qpos_err": qpos_err,
        "ee_pos_err":  ee_pos_err
    }


def cost_function(traj_path: str, max_eps: int = 5):
    ep_names = [d for d in os.listdir(traj_path) if os.path.isdir(os.path.join(traj_path, d))]
    ep_names = ep_names[:max_eps]

    cost = 0.0
    for traj_id in range(len(ep_names)):
        # TODO: check file structure and fix paths if needed
        traj_qpos_actions = np.load(f"{traj_path}/{ep_names[traj_id]}/action_qpos.npy")
        traj_qpos_gt = np.load(f"{traj_path}/{ep_names[traj_id]}/states_qpos.npy")
        traj_ee_gt = np.load(f"{traj_path}/{ep_names[traj_id]}/states_ee.npy")

        res_dict = replay_traj(traj_qpos_actions, traj_qpos_gt, traj_ee_gt)
        cost += np.sum(np.square(res_dict["qpos_err"] + res_dict["ee_pos_err"]))

    return cost


def plot_err(res_dict, ep_name, log_dir):
    qpos_err = res_dict["qpos_err"]
    ee_pos_err = res_dict["ee_pos_err"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot joint errors
    axes[0].plot(qpos_err)
    axes[0].set_title(f"Joint Errors for {ep_name}")
    axes[0].set_ylabel("Error (rad)")
    axes[0].set_xlabel("Time steps")
    axes[0].legend([f"Joint {i}" for i in range(7)], loc='upper right')
    axes[0].grid(True)

    # Plot EE xyz errors
    axes[1].plot(ee_pos_err)
    axes[1].set_title(f"End-Effector XYZ Errors for {ep_name}")
    axes[1].set_ylabel("Error (m)")
    axes[1].set_xlabel("Time steps")
    axes[1].legend(['X', 'Y', 'Z'], loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()

    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{ep_name}.png")
    plt.savefig(plot_path)
    plt.close(fig)
