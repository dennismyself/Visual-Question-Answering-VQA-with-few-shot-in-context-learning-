#!/bin/bash
#SBATCH -J MLMI8_zeroshot
#SBATCH -A MLMI-jq271-SL2-GPU
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
EXP_NAME=Ex2_VQA2_T0-3B_ViT_Mapping_Network_shot_0
### UNCOMMENT BELOW TO USE SBATCH ###
# . /etc/profile.d/modules.sh                # Leave this line (enables the module command)
# module purge                               # Removes all modules still loaded
# module load rhel8/default-amp
# module load cuda/11.1 intel/mkl/2017.4
# source scripts/activate_shared_env.sh
# JOBID=$SLURM_JOB_ID
# LOG=logs/0.$EXP_NAME-log.$JOBID
# ERR=logs/0.$EXP_NAME-err.$JOBID

export OMP_NUM_THREADS=1

## YOUR SCRIPT DOWN HERE
python src/main.py \
    configs/vqa2/zero_shot_vqa_hotpotqa.jsonnet \
    --num_shots 0 \
    --mode test \
    --experiment_name ${EXP_NAME}.$JOBID \
    --accelerator auto \
    --devices 1 \
    --log_prediction_tables \
    --log_prediction_tables_with_images \
    --opts test.batch_size=16 \
        test.load_model_path=/rds/project/rds-xyBFuSj0hm0/MLMI.2022-23/shared/MLMI8/model_checkpoint/mapping_network_on_cc.ckpt \
   >> $LOG 2> $ERR