#!/bin/bash
#SBATCH -J MLMI8_train_cc
#SBATCH -A MLMI-<your_crsid>-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue
#SBATCH -p ampere
#! ############################################################

LOG=/dev/stdout
ERR=/dev/stderr
EXP_NAME=Ex1_finetune_T0_3B_mapping_network_on_conceptual_captions
#UNCOMMENT BELOW FOR SLURM SBATCH
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
module load cuda/11.1 intel/mkl/2017.4
source scripts/activate_shared_env.sh
JOBID=$SLURM_JOB_ID
LOG=logs/$EXP_NAME-log.$JOBID
ERR=logs/$EXP_NAME-err.$JOBID


## YOUR SCRIPT DOWN HERE
python src/main.py \
    configs/conceptual_captions/conceptual_captions.jsonnet \
    --mode train \
    --experiment_name ${EXP_NAME}.$JOBID \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=1 \
    train.batch_size=64 \
    valid.step_size=1 \
    valid.batch_size=64 \
    train.additional.gradient_accumulation_steps=2 \
    train.lr=0.0003 \
   >> $LOG 2> $ERR