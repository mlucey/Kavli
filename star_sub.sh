#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment
#export KERAS_BACKEND=tensorflow
#srun -p cp100 /homes/nramachandra/anaconda3/envs/tf_gpu/bin/python P_MDN.py
srun -p cp100 python P_MDN.py

echo [$SECONDS] End job 


