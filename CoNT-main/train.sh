#!/bin/bash
#DSUB -n CoNTWork
#DSUB -A root.bingxing2.gpuuser356
#DSUB -q root.default
#DSUB -l wuhanG5500
#DSUB --job_type cosched
#DSUB -R 'cpu=12;gpu=2;mem=90000'
#DSUB -N 1
#DSUB -e %J.out 
#DSUB -o %J.out 
module load cuda/11.0.3-gcc-4.8.5-pez
module load anaconda/2021.11
source activate CoNT_py37
python train.py
