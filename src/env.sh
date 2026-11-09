SRC=/home/hj/Alican/training_runs/joint_samplerfix_20261012
HERE=/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_HOME=/home/hj/miniconda3/envs/cuda128 TORCH_CUDA_ARCH_LIST=12.0 PYTORCH_ALLOC_CONF=expandable_segments:True
export GAUSSVERSE_WORKSPACE=/home/hj/Alican GAUSSVERSE_REPO=/home/hj/Alican/gaussverse GAUSSVERSE_DA3_SOURCE=/home/hj/Alican/lyra/Lyra-2/lyra_2/_src/inference/depth_anything_3/src
export PATH=/home/hj/miniconda3/envs/framecrafter/bin:/home/hj/miniconda3/envs/cuda128/bin:$PATH PYTHONPATH=$HERE:/home/hj/Alican:$SRC:/home/hj/Alican/analysis/plan_base_restart_20261008/memorizon_plugins:/home/hj/Alican/training_runs/memory1500_resplat_20260905/python_deps
