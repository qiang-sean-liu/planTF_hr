export PYTHONPATH="/home/users/qiang.liu/wrk/nuplan-devkit-v1.1"
export PYTHONPATH=$(pwd):$PYTHONPATH
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache"

SAVE_DIR="/mnt/nas20/qiang.liu/urbandriver_training_save_dir/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./run_training.py \
  group=$SAVE_DIR \
  +training=training_urban_driver_open_loop_model \
  py_func=train \
  scenario_builder=nuplan \
  cache.cache_path=/mnt/nas20/nuplan_cached/cache_urban_driver_1M \
  data_loader.params.batch_size=32 \
  data_loader.params.num_workers=32 \
  cache.use_cache_without_dataset=true \
  worker=single_machine_thread_pool \
  worker.max_workers=32 \
  optimizer=adam \
  optimizer.lr=1e-4 \
  lightning.trainer.params.max_epochs=30 \
  lr_scheduler=multistep_lr \
  lr_scheduler.milestones='[20]' \
  lr_scheduler.gamma=0.1 \
