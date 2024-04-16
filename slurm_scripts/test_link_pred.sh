#!/bin/bash
#
# Copyright (c) 2024. Luke Braithwaite
# License: MIT
#

srun python link_prediction.py trainer.devices=1 +tags=["${MODEL}","${DATASET}",lp,gnn,exp2,recsys,debug]
