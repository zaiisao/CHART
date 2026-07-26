#!/bin/sh
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
D=/home/sogang/jaehoon/VBPM_reintegration/vbpm_final
cd $D
launch () {  # gpu view seed
  CUDA_VISIBLE_DEVICES=$1 nohup $PY exp1_cut_tempo.py --view $2 --seed $3 \
      > $D/log_$2_s$3.txt 2>&1 &
}
launch 1 cut_tempo 0
launch 1 cut_tempo 1
launch 2 full 0
launch 2 full 1
launch 3 cut_phase 0
launch 3 cut_phase 1
wait
