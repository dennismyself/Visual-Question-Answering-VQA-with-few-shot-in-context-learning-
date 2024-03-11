WDIR=/rds/user/jq271/hpc-work/MLMI8_2022_VQA
BDIR=/rds/project/rds-xyBFuSj0hm0/MLMI.2022-23/shared/MLMI8/
source $WDIR/MLMI-VQA-2022/scripts/activate_shared_env.sh

FILENAMES="Ex4_VQA2_T0-3B_ViT_Mapping_Network_RICES_CAT_a1b1_hotpotqa_shot_4.47131872 

"
for FILENAME in $FILENAMES; do
    EDIR=../Experiments/$FILENAME/test/test_evaluation
    python src/evaluate_from_prediction_file_postprocessing.py $EDIR/predictions.pkl 
done