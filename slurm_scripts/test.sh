#!/bin/bash
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

MODEL_PARAMS=(diag_sheaf bundle_sheaf general_sheaf)
SHEAF_LEARNERS=(node_type_concat edge_type_concat node_type edge_type)
DATASETS=(dblp acm imdb)

N_TRIALS=10
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODEL_PARAMS[@]}
N_SHEAF_LEARNERS=${#SHEAF_LEARNERS[@]}

#IDX=${SLURM_ARRAY_TASK_ID}

for BLOCK_IDX in {0..35}
do
  NUM_OPTIONS=$((N_DATASETS * N_SHEAF_LEARNERS))
  MODEL_IDX=$((BLOCK_IDX / NUM_OPTIONS))
  DATA_SHEAF_IDX=$((BLOCK_IDX % NUM_OPTIONS))
  SHEAF_LEARNER_IDX=$((DATA_SHEAF_IDX / N_DATASETS))
  DATA_IDX=$((DATA_SHEAF_IDX % N_DATASETS))

  MODEL=${MODEL_PARAMS[MODEL_IDX]}
  DATASET=${DATASETS[DATA_IDX]}
  SHEAF_LEARNER=${SHEAF_LEARNERS[SHEAF_LEARNER_IDX]}
  echo "($BLOCK_IDX)" "$MODEL" "$SHEAF_LEARNER" "$DATASET"
done