import argparse
import sys
import numpy as np
import torch
from datetime import datetime
import cma

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import set_sim_config
from realm.utils import cost_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--max_eps', type=int, required=False, default=5)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/app/logs/replay_trajectory/{run_id}"
    task_cfg_path = f"IMPACT/trajectory_replay/default.yaml"
    rendering_mode = "r"
    traj_path = "/app/data/droid_1.0.1"

    set_sim_config(rendering_mode=rendering_mode)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=["Default"],
        rendering_mode=rendering_mode,
        robot="DROID_no_wrist_cam",
        no_rendering=True
    )

    friction = np.array([0.05, 0.15, 0.25, 0.15, 0.75, 0.15, 0.50])
    armature = np.array([0.50, 0.20, 0.50, 0.20, 0.25, 0.00, 0.25])
    initial_flat_params = np.concatenate((friction, armature))

    # sigma0 is the initial step-size (standard deviation) for the search.
    # A good starting point is often 0.1 to 1.0, depending on the scale of your variables.
    initial_sigma = 0.075

    # --- 4. Define CMA-ES options (optional but recommended) ---
    # See https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html#cma.evolution_strategy.CMAOptions
    # for a full list of options.
    options = {
        'seed': 42,  # For reproducibility
        'maxfevals': 500,  # Maximum number of function evaluations
        'popsize': 100,  # Population size (number of solutions evaluated per iteration)
        'tolfun': 1e-6,  # Tolerance for cost function value
        'tolx': 1e-6,  # Tolerance for variable changes
        'verb_disp': 1,  # Display progress every X iterations
        'verbose': -9,  # Reduce verbosity of intermediate output
        'bounds': [0, 2]  # Example for boundary constraints on *each* flattened parameter
    }

    # --- 5. Run the optimization ---
    def replay_error():
        return cost_function(env=env, traj_path=traj_path, max_eps=args.max_eps)

    es = cma.fmin(
        replay_error,
        initial_flat_params,
        initial_sigma,
        options=options
    )

    # --- 6. Access the results ---
    best_flat_params = es[0]  # Best solution found
    best_cost = es[1]  # Cost value of the best solution
    num_evaluations = es[2]  # Number of function evaluations
    num_iterations = es[3]  # Number of iterations
    mean_params_flat = es[5]  # Mean of the population (often a good solution)

    print(options)

    print("\n--- Optimization Results ---")
    print(f"Best cost found: {best_cost}")
    print(f"Number of function evaluations: {num_evaluations}")
    print(f"Number of iterations: {num_iterations}")

    # Unflatten the best parameters to get them back in their original array forms
    optimal_friction = best_flat_params[:7]
    optimal_armature = best_flat_params[7:14]

    print("\nOptimized friction:\n", optimal_friction)
    print("\nOptimized armature:\n", optimal_armature)

    og.shutdown()
    sys.exit(0)
