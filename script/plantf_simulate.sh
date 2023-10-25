#cwd="/mnt/nas20/qiang.liu/plantf_training_save_dir/training/planTF/2023.09.26.15.01.11"
#cwd="/mnt/nas20/qiang.liu/plantf_training_save_dir/training/planTF/2023.10.05.15.41.17"
#cwd="/mnt/nas20/qiang.liu/debug_training_save_dir/training/planTF/2023.10.06.05.27.39"
#cwd="/mnt/nas20/qiang.liu/plantf_training_save_dir/plantf_augmentOff_stateAttnEncOff_lstm/training/planTF/2023.10.08.02.56.12"
#cwd="/mnt/nas20/qiang.liu/plantf_training_save_dir/plantf_stateAttnEncOn_lstm/training/planTF/2023.10.09.12.30.03"
cwd="/mnt/nas20/qiang.liu/plantf_training_save_dir/plantf_stateAttOn_natOn_posEncOff/training/planTF/2023.10.13.01.14.25"
CKPT_ROOT="$cwd/checkpoints"
EXPERIMENT="plantf_stateAttOn_natOn_posEncOff"

SAVE_DIR="/mnt/nas20/qiang.liu/plantf_simulation_save_dir"

export PYTHONPATH="/home/users/qiang.liu/wrk/nuplan-devkit-v1.1"
export PYTHONPATH=$(pwd):$PYTHONPATH
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export NUPLAN_DATA_ROOT="/mnt/nas20"  # created symbolic link under /mnt/nas20/nuplanv1.1/test
export LD_LIBRARY_PATH="/home/users/qiang.liu/miniconda3/envs/nuplan/lib/python3.9/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export TMPDIR="/mnt/nas20/qiang.liu/ray_temp_dir"

PLANNER="planTF"
SPLIT="test14-random"
CHALLENGES="closed_loop_nonreactive_agents"

for challenge in $CHALLENGES; do
    python run_simulation.py \
        experiment_name=$EXPERIMENT \
        group=$SAVE_DIR \
        +simulation=$challenge \
        planner=planTF \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=20 \
        experiment_uid=$SPLIT/planTF \
        verbose=true \
        planner.imitation_planner.planner_ckpt="$CKPT_ROOT/last.ckpt" \
        exit_on_failure=True
done