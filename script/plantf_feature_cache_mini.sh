export PYTHONPATH="/home/users/qiang.liu/wrk/nuplan-devkit-v1.1"
export PYTHONPATH=$(pwd):$PYTHONPATH
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache"

echo $PYTHONPATH

python run_training.py \
    py_func=cache +training=train_planTF \
    scenario_builder=nuplan \
    cache.cache_path=/mnt/nas20/nuplan_cached/cache_plantf_mini \
    cache.cleanup_cache=true \
    scenario_filter=mini \
    worker.threads_per_node=40
