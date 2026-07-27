#!/bin/bash
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
D=/home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq
cd $D
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=1 $PY $D/innovq.py --placement pf_sm101 --seed $s --pre 600 --steps 800 --beta_ramp 100 --every 50 > $D/cell_pf_sm101_s$s.log 2>&1
done
touch $D/DONE_gpu1
