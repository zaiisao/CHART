#!/bin/sh
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
D=/home/sogang/jaehoon/VBPM_reintegration/vbpm_final
cd $D
CUDA_VISIBLE_DEVICES=1 $PY exp1_deploy.py --view cut_tempo --seed 0 > $D/dep_cut_tempo_s0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY exp1_deploy.py --view cut_tempo --seed 1 > $D/dep_cut_tempo_s1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY exp1_deploy.py --view full --seed 0 > $D/dep_full_s0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY exp1_deploy.py --view full --seed 1 > $D/dep_full_s1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY exp1_deploy.py --view cut_phase --seed 0 > $D/dep_cut_phase_s0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY exp1_deploy.py --view cut_phase --seed 1 > $D/dep_cut_phase_s1.txt 2>&1 &
wait
