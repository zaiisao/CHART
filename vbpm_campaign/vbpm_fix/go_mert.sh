#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd /home/sogang/jaehoon/VBPM_reintegration
exec /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python vbpm_fix/run_mert.py \
  --steps 1200 --warmup 600 --frames 256 --bs 16 --n_eval 25 --max_frames 1200 --K 300 --n_shift 6
