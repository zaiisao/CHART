#!/bin/bash
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
cd /home/sogang/jaehoon/VBPM_reintegration/vbpm_innovq
i=0
for gp in 0.00055 0.06; do for fz in 0 1; do for tg in 0 1; do for rec in bce hybrid; do
  nm="F_g$( [ $gp = 0.06 ] && echo hi || echo lo)_f${fz}_t${tg}_${rec}"
  args="--recon $rec --placement cold --crop 1500 --fast --no_abort --gamma_phase $gp"
  [ $fz = 1 ] && args="$args --dec_pre 600 --dec_freeze"
  [ $tg = 1 ] && args="$args --tg"
  g=$((i % 4))
  CUDA_VISIBLE_DEVICES=$g $PY -u innovq.py $args --tag $nm --seed 0 --pre 0 --steps 800 --every 200 \
     > logs/$nm.log 2>&1 &
  i=$((i+1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done; done; done; done
wait
