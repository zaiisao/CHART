#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
cd /home/sogang/jaehoon/VBPM_reintegration
exec /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python vbpm_fix/run_dirac.py   --only base --steps 800 --warmup 400 --n_eval 30 --max_frames 1600 --K 400 --n_shift 8   --out /home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/results_dirac_base.json
