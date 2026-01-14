import streamlit as st
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import dashboard_utils
from collections import defaultdict

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

# Define the logs directory
LOGS_DIR = "logs"

# Initialize session state for selection
if "selected_experiment" not in st.session_state:
    st.session_state.selected_experiment = None

def render_tree(path, depth=0):
    """Recursive function to render the directory tree using expanders."""
    if depth > 5:
        return

    subdirs = dashboard_utils.get_subdirectories(path)
    if not subdirs:
        return

    for d in subdirs:
        full_path = os.path.join(path, d)
        is_exp = dashboard_utils.is_experiment_folder(full_path)

        # Unique key for widgets
        key_base = full_path

        sub_subdirs = dashboard_utils.get_subdirectories(full_path)
        has_children = len(sub_subdirs) > 0

        if has_children:
            label = f"📁 {d}" if not is_exp else f"🔬 {d}"

            with st.sidebar.expander(label, expanded=False):
                # If this node itself is selectable
                if is_exp:
                    if st.button(f"👉 Select {d}", key=f"btn_{key_base}_inner"):
                        st.session_state.selected_experiment = full_path

                # Recurse
                render_tree(full_path, depth + 1)
        else:
            # Leaf node
            if is_exp:
                # Leaf experiment button
                if st.sidebar.button(f"🔬 {d}", key=f"btn_{key_base}_leaf"):
                    st.session_state.selected_experiment = full_path
            else:
                 # It's a folder but has no subdirectories and isn't an experiment?
                 st.sidebar.markdown(f"📁 {d}")

# Sidebar
st.sidebar.title("Experiment Browser")
render_tree(LOGS_DIR)

# Sidebar Filter
st.sidebar.markdown("---")
st.sidebar.header("Filter Videos")
selected_filters = []

# Filter UI
# "Each task and its perturbations should have a checkbox, that if ticked, shows only relevant videos."
# We iterate tasks and create expanders
for task in dashboard_utils.SUPPORTED_TASKS:
    with st.sidebar.expander(task, expanded=False):
        # "All" checkbox for convenience
        all_key = f"filter_all_{task}"
        select_all = st.checkbox(f"All Perturbations", key=all_key)

        for pert in dashboard_utils.SUPPORTED_PERTURBATIONS:
            pert_key = f"filter_{task}_{pert}"
            is_checked = st.checkbox(pert, key=pert_key)

            if select_all or is_checked:
                selected_filters.append((task, pert))


# Main Content
if st.session_state.selected_experiment and os.path.exists(st.session_state.selected_experiment):
    selected_path = st.session_state.selected_experiment

    # Header Parsing
    rel_path = os.path.relpath(selected_path, LOGS_DIR)
    path_parts = rel_path.split(os.sep)

    experiment_name = path_parts[0] if len(path_parts) > 0 else "N/A"
    model_name = path_parts[1] if len(path_parts) > 1 else "N/A"
    run_id = path_parts[2] if len(path_parts) > 2 else "N/A"

    st.title("Experiment Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiment", experiment_name)
    c2.metric("Model", model_name)
    c3.metric("Run ID", run_id)

    st.divider()

    # Load Reports
    df = dashboard_utils.load_reports(selected_path)

    # Plots Section
    st.header("Plots")
    try:
        if df is not None and not df.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Success Rate per Perturbation")
                # Clean perturbation names if needed (remove ['...'])
                df['clean_pert'] = df['perturbation'].apply(lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x))

                # Group by perturbation and calculate mean success
                if 'binary_SR' in df.columns:
                    pert_sr = df.groupby('clean_pert')['binary_SR'].mean().reset_index()

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(pert_sr['clean_pert'], pert_sr['binary_SR'], color='skyblue')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Success Rate")
                    ax.set_xlabel("Perturbation")
                    plt.xticks(rotation=45, ha='right')

                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    st.image(buf)
                else:
                    st.info("No binary_SR column found for plots.")

            with c2:
                st.subheader("Success Rate per Task")
                if 'binary_SR' in df.columns:
                    task_sr = df.groupby('task')['binary_SR'].mean().reset_index()

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(task_sr['task'], task_sr['binary_SR'], color='lightgreen')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Success Rate")
                    ax.set_xlabel("Task")
                    plt.xticks(rotation=45, ha='right')

                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    st.image(buf)
        else:
            st.info("No data available for plots.")
    except Exception as e:
        st.error(f"Error in Plots: {e}")

    # Experiment Status Section
    st.header("Experiment Status")
    try:
        # Resolve experiment directory from selected_path
        rel_path = os.path.relpath(selected_path, LOGS_DIR)
        parts = rel_path.split(os.sep)

        if len(parts) > 0:
            experiment_name = parts[0]
            experiment_path = os.path.join(LOGS_DIR, experiment_name)

            metadata, err = dashboard_utils.load_experiment_metadata(experiment_path)

            if metadata:
                tasks_indices = metadata.get("task_ids", [])
                perts_indices = metadata.get("perturbation_ids", [])
                required_repeats = metadata.get("repeats", 0)

                st.write(f"**Target Configuration (from {experiment_name}/metadata.json):** Tasks: {tasks_indices}, Perturbations: {perts_indices}, Repeats: {required_repeats}")

                status, msg = dashboard_utils.check_experiment_status(df, tasks_indices, perts_indices, required_repeats)

                if status:
                    st.success("✅ " + msg)
                else:
                    st.error("❌ " + msg)

                # Show completed combinations
                completed = dashboard_utils.get_completed_experiments(df, required_repeats)
                if completed:
                    with st.expander("Completed Combinations", expanded=True):
                        grouped = defaultdict(list)
                        for t, p in completed:
                            grouped[t].append(p)

                        for t, perts in grouped.items():
                            st.write(f"- **{t}**: {', '.join(perts)}")
            else:
                st.warning(err)
        else:
            st.warning("Could not determine experiment directory.")

    except Exception as e:
        st.error(f"Error in Experiment Status: {e}")

    # Aggregated Reports Section
    st.header("Aggregated Reports")
    try:
        if df is not None:
            st.dataframe(df, height=300)
        else:
            st.info("No reports found.")
    except Exception as e:
        st.error(f"Error in Aggregated Reports: {e}")

    # Videos Section
    st.header("Videos")
    videos = dashboard_utils.get_videos(selected_path)

    # Filter videos
    filtered_videos = dashboard_utils.filter_videos(videos, selected_filters)

    if filtered_videos:
        # Tiled viewer with 3 columns
        cols = st.columns(3)
        for i, video_path in enumerate(filtered_videos):
            with cols[i % 3]:
                st.video(video_path)
                st.caption(os.path.basename(video_path))
    else:
        if videos:
            st.info("No videos match the selected filters.")
        else:
            st.info("No videos found.")
else:
    if not os.path.exists(LOGS_DIR):
         st.error(f"Logs directory '{LOGS_DIR}' not found.")
    else:
        st.info("Please select an experiment from the sidebar.")
