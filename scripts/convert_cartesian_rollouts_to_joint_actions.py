from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONTROL_DT = 1.0 / 15.0


class JointActionConverter:
    def __init__(self) -> None:
        from realm.robots.robot_ik.robot_ik_solver import RobotIKSolver

        self._solver = RobotIKSolver()

    def cartesian_velocity_to_joint_position(
        self,
        cartesian_velocity: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> np.ndarray:
        joint_velocity_cmd = self._solver.cartesian_velocity_to_joint_velocity(
            cartesian_velocity,
            robot_state={
                "joint_positions": joint_position,
                "joint_velocities": joint_velocity,
            },
        )
        joint_delta = self._solver.joint_velocity_to_delta(joint_velocity_cmd)
        return np.asarray(joint_position, dtype=np.float64) + np.asarray(joint_delta, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy rollout folders and rewrite frames.jsonl actions from "
            "[6D cartesian velocity + gripper] to [7D joint position target + gripper]."
        )
    )
    parser.add_argument("input_root", type=Path, help="Root folder containing rollout subfolders.")
    parser.add_argument("output_root", type=Path, help="New root folder to write converted rollouts into.")
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_CONTROL_DT,
        help="Time step used to estimate joint velocities from robot_state finite differences. Default: 1/15.",
    )
    parser.add_argument(
        "--joint-velocity-mode",
        choices=("finite-difference", "zero"),
        default="finite-difference",
        help="How to estimate the current joint velocity needed by the IK solver.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_root first if it already exists.",
    )
    return parser.parse_args()


def validate_paths(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    if input_root.resolve() == output_root.resolve():
        raise ValueError("input_root and output_root must be different directories.")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)


def discover_frames_files(input_root: Path) -> list[Path]:
    frames_files = sorted(input_root.rglob("frames.jsonl"))
    if not frames_files:
        raise FileNotFoundError(f"No frames.jsonl files found under: {input_root}")
    return frames_files


def load_frames(frames_path: Path) -> list[dict]:
    frames: list[dict] = []
    with frames_path.open("r", encoding="utf-8") as frames_file:
        for line_number, line in enumerate(frames_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frames.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {frames_path} at line {line_number}") from exc
    return frames


def estimate_joint_velocities(
    joint_positions: np.ndarray,
    dt: float,
    mode: str,
) -> np.ndarray:
    if mode == "zero" or len(joint_positions) <= 1:
        return np.zeros_like(joint_positions, dtype=np.float64)

    joint_velocities = np.zeros_like(joint_positions, dtype=np.float64)
    joint_velocities[0] = (joint_positions[1] - joint_positions[0]) / dt
    joint_velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) / dt
    return joint_velocities


def convert_frames(
    frames: list[dict],
    converter: JointActionConverter,
    dt: float,
    joint_velocity_mode: str,
) -> list[dict]:
    if not frames:
        return []

    joint_positions = np.asarray([frame["robot_state"] for frame in frames], dtype=np.float64)
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 7:
        raise ValueError(
            "Expected every frame['robot_state'] to be a 7D joint position vector, "
            f"got shape {joint_positions.shape}."
        )

    joint_velocities = estimate_joint_velocities(joint_positions, dt=dt, mode=joint_velocity_mode)
    converted_frames: list[dict] = []

    for index, frame in enumerate(frames):
        action = np.asarray(frame["action"], dtype=np.float64)
        if action.ndim != 1 or action.shape[0] != 7:
            raise ValueError(
                "Expected every frame['action'] to be a 7D vector "
                "([6D cartesian velocity, 1D gripper]), "
                f"got shape {action.shape} at frame index {frame.get('index', index)}."
            )

        joint_position_target = converter.cartesian_velocity_to_joint_position(
            cartesian_velocity=action[:6],
            joint_position=joint_positions[index],
            joint_velocity=joint_velocities[index],
        )
        converted_action = np.concatenate([joint_position_target, action[-1:]])

        converted_frame = dict(frame)
        converted_frame["action"] = converted_action.astype(np.float32).tolist()
        converted_frames.append(converted_frame)

    return converted_frames


def write_frames(frames_path: Path, frames: list[dict]) -> None:
    with frames_path.open("w", encoding="utf-8") as frames_file:
        for frame in frames:
            frames_file.write(json.dumps(frame, ensure_ascii=False) + "\n")


def copy_tree(input_root: Path, output_root: Path) -> None:
    shutil.copytree(input_root, output_root)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    validate_paths(input_root, output_root, overwrite=args.overwrite)
    frames_files = discover_frames_files(input_root)
    copy_tree(input_root, output_root)

    converter = JointActionConverter()
    converted_rollouts = 0
    converted_frames = 0

    for source_frames_path in frames_files:
        target_frames_path = output_root / source_frames_path.relative_to(input_root)
        frames = load_frames(source_frames_path)
        rewritten_frames = convert_frames(
            frames,
            converter=converter,
            dt=args.dt,
            joint_velocity_mode=args.joint_velocity_mode,
        )
        write_frames(target_frames_path, rewritten_frames)
        converted_rollouts += 1
        converted_frames += len(rewritten_frames)
        print(f"Converted {source_frames_path.relative_to(input_root)} ({len(rewritten_frames)} frames)")

    print(
        f"Finished converting {converted_rollouts} rollouts and {converted_frames} frames "
        f"from {input_root} to {output_root}."
    )


if __name__ == "__main__":
    main()