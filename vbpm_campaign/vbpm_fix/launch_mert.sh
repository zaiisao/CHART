#!/bin/bash
cd /home/sogang/jaehoon/VBPM_reintegration
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
L=/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/logs
mkdir -p $L
run () { gpu=$1; tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY vbpm_fix/run_variant_a.py --mode mert \
    --steps 1200 --warmup 600 --frames 512 --bs 16 --eval_every 300 --n_eval 30 --tag $tag "$@" \
    > $L/$tag.log 2>&1 &
}
run 1 M_base --baseline 1
run 2 M_c02  --corr_scale 0.2  --tempo_init 1
run 3 M_c05  --corr_scale 0.5  --tempo_init 1
run 3 M_cpi  --corr_scale 3.14159265 --tempo_init 1
wait
