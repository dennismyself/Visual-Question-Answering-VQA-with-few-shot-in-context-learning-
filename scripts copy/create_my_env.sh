#!/bin/bash
module purge
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
module load slurm

eval "$(conda shell.bash hook)"
conda create --clone /rds/project/rds-xyBFuSj0hm0/MLMI.2022-23/shared/MLMI8/vqa_env -p /home/$USER/rds/hpc-work/conda_env/mlmi_2022_vqa
