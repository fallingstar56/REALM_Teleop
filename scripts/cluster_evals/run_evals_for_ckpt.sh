#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <checkpoint_name>"
  exit 1
fi

CHECKPOINT_NAME=$1
POLICY_CONFIG="$2"
CHECKPOINT_PATH="$3"
POLICY_RUN_DIR="$4"
BASE_PORT="$5"
EXPERIMENT_NAME="minimal_1600"
RUN_ID=$(date +%Y%m%d_%H%M%S)


# This script takes a checkpoint name as an argument and runs sbatch for integers 0-7.
for i in {0..7}
do
  if [[ "$i" -eq 2 ]] || [[ "$i" -eq 3 ]] || [[ "$i" -eq 5 ]]; then
    continue
  fi
    sbatch --array=0,1,2,3,4,11,13 scripts/cluster_evals/run_single_eval.sh \
      $i \
      25 \
      800 \
      "$CHECKPOINT_NAME" \
      "$POLICY_CONFIG" \
      "$CHECKPOINT_PATH" \
      "$BASE_PORT" \
      "$EXPERIMENT_NAME" \
      "$RUN_ID" \
      "$POLICY_RUN_DIR"
done
