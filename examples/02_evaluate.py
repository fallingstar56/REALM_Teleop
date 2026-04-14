import argparse
import os
from realm.eval import evaluate
import sys

import omnigibson as og

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--perturbation_id', type=int, required=False, default=0)
    parser.add_argument('--task_id', type=int, required=False, default=0)
    parser.add_argument('--repeats', type=int, required=False, default=5)
    parser.add_argument('--max_steps', type=int, required=False, default=500)
    parser.add_argument('--horizon', type=int, required=False, default=8)
    parser.add_argument('--task_cfg_path', type=str, required=False, default=None)
    parser.add_argument('--model_name', type=str, required=False, default=None)
    parser.add_argument('--model_type', type=str, required=False, default=None)
    parser.add_argument('--port', type=int, required=False, default=8000)
    parser.add_argument('--host', type=str, required=False, default="127.0.0.1", help='Inference server host')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=False, default=None)
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--rendering_mode', type=str, required=False, default=None, help='Omnigibson rendering mode (pt, rt, r)')
    parser.add_argument('--multi-view', action='store_true', help='Enable second external camera')
    parser.add_argument('--resume', action='store_true', help='Resume from existing run report if found')
    parser.add_argument('--no_record', action='store_true', help='Do not record videos from runs.')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering completely')
    parser.add_argument('--robot', type=str, required=False, default="DROID", help='Robot type')
    parser.add_argument('--action_source', type=str, choices=["policy", "teleop"], default="policy", help='Action source for rollout control')
    args = parser.parse_args()

    assert args.experiment_name is not None
    if args.action_source == "policy":
        assert args.model_name is not None
        assert args.model_type is not None
    else:
        if args.model_name is None:
            args.model_name = "vr_teleop"
        if args.model_type is None:
            args.model_type = "teleop"
    #assert not (args.task_cfg_path and args.task_id), f"Either task --task_cfg_path or --task_id should be specified, but not both."

    base_log_dir = args.log_dir if args.log_dir is not None else "/app/logs"
    experiment_root_dir = os.path.join(base_log_dir, args.experiment_name)
    log_dir = os.path.join(experiment_root_dir, args.model_name)
    if args.run_id is not None:
        log_dir = os.path.join(log_dir, args.run_id)

    evaluate(
        task_id=args.task_id,
        perturbation_id=args.perturbation_id,
        repeats=args.repeats,
        max_steps=args.max_steps,
        horizon=args.horizon,
        model_type=args.model_type,
        port=args.port,
        host=args.host,
        log_dir=log_dir,
        multi_view=args.multi_view,
        resume=args.resume,
        no_record=args.no_record,
        no_render=args.no_render,
        rendering_mode=args.rendering_mode,
        task_cfg_path=args.task_cfg_path,
        robot=args.robot,
        action_source=args.action_source,
        experiment_name=args.experiment_name,
        experiment_root_dir=experiment_root_dir,
    )
    og.shutdown()
    sys.exit(0)
