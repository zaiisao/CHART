#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
cd /home/sogang/jaehoon/VBPM_reintegration
exec /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python vbpm_fix/final_eval_b.py --mode dirac --cap 1600 --K 400 --roll 1000 --n_roll 25
