#!/bin/bash
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

MODELS=( AllDeepSets AllSetsTransformer HCHA HGNN SheafHyperGNN SheafHyperGNN-TE SheafHyperGNN-ensemble )
DATASETS=( DeepDTNet KEGG )

N_SEEDS=10
N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODELS[@]}

for IDX in {0..139}
do
N_RUN=$(( IDX / N_SEEDS ))
MODEL_IDX=$(( N_RUN / N_DATASETS ))
DATA_IDX=$(( N_RUN % N_DATASETS ))

MODEL=${MODELS[MODEL_IDX]}
DATASET=${DATASETS[DATA_IDX]}
SPLIT=$(( IDX % N_SEEDS ))
  echo "($IDX)" "$MODEL" "$DATASET" "$SPLIT"
done