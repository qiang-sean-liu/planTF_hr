export PYTHONPATH="/home/users/qiang.liu/wrk/nuplan-devkit-v1.1"
export PYTHONPATH=$(pwd):$PYTHONPATH
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache"

SAVE_DIR="/mnt/nas20/qiang.liu/debug_training_save_dir/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  group=$SAVE_DIR \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/mnt/nas20/nuplan_cached/cache_plantf_mini cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=20 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=15 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  #wandb.mode=online wandb.project=nuplan wandb.name=plantf