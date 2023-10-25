cwd="/mnt/nas20/qiang.liu/urbandriver_training_save_dir/training/urban_driver_open_loop_model/2023.10.04.02.26.13"
CKPT_ROOT="$cwd/checkpoints"
EXPERIMENT="default"

SAVE_DIR="/mnt/nas20/qiang.liu/plantf_simulation_save_dir"

export PYTHONPATH="/home/users/qiang.liu/wrk/nuplan-devkit-v1.1"
export PYTHONPATH=$(pwd):$PYTHONPATH
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export NUPLAN_DATA_ROOT="/mnt/nas20"  # created symbolic link under /mnt/nas20/nuplanv1.1/test
export LD_LIBRARY_PATH="/home/users/qiang.liu/miniconda3/envs/nuplan/lib/python3.9/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export TMPDIR="/mnt/nas20/qiang.liu/ray_temp_dir"

PLANNER="urban_driver_open_loop"
SPLIT="test14-random"
CHALLENGES="closed_loop_nonreactive_agents"

for challenge in $CHALLENGES; do
    python run_simulation.py \
        experiment_name=$EXPERIMENT \
        group=$SAVE_DIR \
        +simulation=$challenge \
        planner=ml_planner \
        model=urban_driver_open_loop_model \
        'planner.ml_planner.model_config=${model}' \
        scenario_builder=nuplan_challenge \
        scenario_filter=$SPLIT \
        worker.threads_per_node=20 \
        experiment_uid=$SPLIT/$planner \
        verbose=true \
        planner.ml_planner.checkpoint_path="$CKPT_ROOT/last.ckpt"
done