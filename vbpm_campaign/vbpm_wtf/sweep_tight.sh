#!/bin/bash
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
cd /home/sogang/jaehoon/VBPM_reintegration/vbpm_wtf
for rho in 0.95 0.99 0.999; do
for slv in 0.002 0.01; do
for al in 1.0 3.0; do
  echo "### rho=$rho s_lv=$slv alpha=$al"
  CUDA_VISIBLE_DEVICES=1 $PY q4_probe4_pf.py --split train --skip 100 --cap 118 \
     --rho $rho --s_lv $slv --alpha $al --nbins 72 --cells "sup|tight" \
     --out sweep_r${rho}_s${slv}_a${al}.json 2>&1 | grep -E "sup\|tight|eval songs"
done; done; done
