#!/bin/bash
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

N_SEEDS=10
DATASETS=( DeepDTNet KEGG )
MAP_TYPES=( DiagSheafs GeneralSheafs OrthoSheafs LowRankSheafs )

N_DATASETS=${#DATASETS[@]}
N_FEATS=${#MAP_TYPES[@]}

#IDX=${SLURM_ARRAY_TASK_ID}
#N_RUN=$(( IDX / N_SEEDS ))
#FEAT_IDX=$(( N_RUN / N_DATASETS ))
#DATA_IDX=$(( N_RUN % N_DATASETS ))
#
#SHEAF_TYPE=${MAP_TYPES[FEAT_IDX]}
#DATASET=${DATASETS[DATA_IDX]}
#SPLIT=$(( IDX % N_TRIALS ))

for IDX in {0..79}
do
N_RUN=$(( IDX / N_SEEDS ))
MODEL_IDX=$(( N_RUN / N_DATASETS ))
DATA_IDX=$(( N_RUN % N_DATASETS ))

MODEL=${MAP_TYPES[MODEL_IDX]}
DATASET=${DATASETS[DATA_IDX]}
SPLIT=$(( IDX % N_SEEDS ))
  echo "($IDX)" "$MODEL" "$DATASET" "$SPLIT"
done