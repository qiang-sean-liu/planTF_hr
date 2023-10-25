SIMULATION_DIR='[/mnt/nas20/qiang.liu/plantf_simulation_save_dir/default/closed_loop_nonreactive_agents/test14-random/planTF]'
EXPERIMENT="enc3_ignoreEgoHist_imiObj_withAgents_mlpDecoder_withGoalEmb_ckpt19"

SAVE_DIR="/mnt/nas20/qiang.liu/nuplan_simulation_save_dir"

export PYTHONPATH="/home/users/qiang.liu/wrk/nuplan-devkit-v1.1"
export PYTHONPATH=$(pwd):$PYTHONPATH

export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache"

# to deal with protobuf
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HYDRA_FULL_ERROR=1

python ../nuplan-devkit-v1.1/nuplan/planning/script//run_nuboard.py \
    scenario_builder=nuplan_mini \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT \
    simulation_path=$SIMULATION_DIR \
    port_number=5008
