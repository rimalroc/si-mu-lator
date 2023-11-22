#!/bin/bash

#SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif
SING_IMG=/Data/images/hls4ml_sandbox
singularity exec -B /Data -B /afs ${SING_IMG} python make_small_dataset.py