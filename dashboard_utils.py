import os
import glob
import pandas as pd
import json

SUPPORTED_TASKS = [
    "put_green_block_in_bowl", #0
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
    'V-VIEW', # 2
    'V-SC', # 3
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'SB-NOUN', # 11
    'SB-VRB', # 12
    'VB-POSE', # 13
    'VB-MOBJ', # 14
    'VSB-NOBJ' # 15
]

def get_subdirectories(path):
    if not os.path.exists(path):
        return []
    try:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except OSError:
        return []

def is_experiment_folder(path):
    """Check if the folder contains 'reports' or 'videos' subdirectories."""
    return os.path.isdir(os.path.join(path, "reports")) or os.path.isdir(os.path.join(path, "videos"))

def load_reports(experiment_path):
    reports_path = os.path.join(experiment_path, "reports")
    if not os.path.exists(reports_path):
        return None

    csv_files = glob.glob(os.path.join(reports_path, "*.csv"))
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            # We can't log to streamlit here directly without passing st, but we can print or ignore
            print(f"Error reading {f}: {e}")

    if not dfs:
        return None

    try:
        aggregated_df = pd.concat(dfs, axis=0, ignore_index=True)
        return aggregated_df
    except Exception as e:
        print(f"Error aggregating CSVs: {e}")
        return None

def get_videos(experiment_path):
    videos_path = os.path.join(experiment_path, "videos")
    if not os.path.exists(videos_path):
        return []
    return sorted(glob.glob(os.path.join(videos_path, "*.mp4")))

def load_experiment_metadata(experiment_path):
    """Loads metadata.json from the experiment directory."""
    metadata_path = os.path.join(experiment_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None, f"Metadata file not found at {metadata_path}"

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata, None
    except Exception as e:
        return None, f"Error reading metadata: {e}"

def check_experiment_status(df, tasks_indices, perts_indices, required_repeats):
    if df is None:
        return False, "No data loaded."

    target_tasks = []
    for i in tasks_indices:
        if 0 <= i < len(SUPPORTED_TASKS):
            target_tasks.append(SUPPORTED_TASKS[i])

    target_perts = []
    for i in perts_indices:
        if 0 <= i < len(SUPPORTED_PERTURBATIONS):
            # Format to match CSV string representation of list
            target_perts.append(f"['{SUPPORTED_PERTURBATIONS[i]}']")

    if not target_tasks or not target_perts:
        return False, "Invalid task or perturbation indices in experiment name."

    # Check existence
    all_good = True
    missing = []

    for t in target_tasks:
        for p in target_perts:
            # Filter df
            count = len(df[(df['task'] == t) & (df['perturbation'] == p)])
            if count < required_repeats:
                all_good = False
                missing.append(f"{t} | {p}: Found {count}/{required_repeats}")

    if all_good:
        return True, "All required tasks and perturbations evaluated with sufficient samples."
    else:
        return False, "Missing evaluations:\n" + "\n".join(missing)

def get_completed_experiments(df, required_repeats=1):
    """
    Returns a list of tuples (task, perturbation) that have >= required_repeats in the dataframe.
    """
    if df is None or df.empty:
        return []

    completed = []
    # We iterate over all supported combinations to check them?
    # Or simpler: group by task and perturbation in df and check counts.

    # Group by 'task' and 'perturbation' and count
    try:
        counts = df.groupby(['task', 'perturbation']).size()

        for (task, pert_str), count in counts.items():
            if count >= required_repeats:
                # pert_str in CSV is "['PERT']", we need to clean it?
                # The dataframe loaded from CSV has "['Default']" as string.
                # We should extract 'Default' from it.
                pert_clean = pert_str.replace("['", "").replace("']", "")
                completed.append((task, pert_clean))

        # Sort for display stability
        completed.sort()
    except Exception as e:
        print(f"Error in get_completed_experiments: {e}")
        return []

    return completed

def filter_videos(video_paths, selected_filters):
    """
    Filters video paths based on selected (task, perturbation) tuples.
    selected_filters: list of (task, perturbation) tuples.
    If selected_filters is empty, returns all videos?
    Wait, logic decision: "if ticked, shows only relevant videos".
    If nothing ticked, assume show all.
    """
    if not selected_filters:
        return video_paths

    filtered = []
    for v in video_paths:
        filename = os.path.basename(v)
        # Check if video matches ANY of the selected filters
        match = False
        for task, pert in selected_filters:
            # Check if task and pert are in filename.
            # Filename format: timestamp_model_rollout_TASK_PERT_runid.mp4
            # We need robust checking because TASK might be substring of another TASK (unlikely given names)
            # but PERT definitely can overlap?
            # SUPPORTED_TASKS are distinct enough.

            if task in filename:
                # Now check perturbation.
                # 'V-AUG' vs 'Default'
                # If filename contains 'V-AUG', it matches.
                # Note: filename might not have brackets.
                # Logging: f"{timestamp}_{model}_rollout_{task}_{perturbations[0]}_{run_id}"
                # perturbation is just the string, e.g. "V-AUG".

                # Boundary check?
                # e.g. "put_green_block_in_bowl" matches.
                # "V-AUG" matches.

                if pert in filename:
                    match = True
                    break

        if match:
            filtered.append(v)

    return filtered
