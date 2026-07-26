#!/bin/bash
cd /home/sogang/jaehoon/VBPM_reintegration
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
L=/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/logs
mkdir -p $L
run () { # gpu tag args...
  gpu=$1; tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY vbpm_fix/run_variant_a.py --mode dirac \
    --steps 800 --warmup 400 --eval_every 200 --n_eval 30 --tag $tag "$@" \
    > $L/$tag.log 2>&1 &
}
run 1 D_base      --baseline 1
run 1 D_tinit     --corr_scale 0.0  --tempo_init 1
run 1 D_c005      --corr_scale 0.05 --tempo_init 1
run 2 D_c02       --corr_scale 0.2  --tempo_init 1
run 2 D_c05       --corr_scale 0.5  --tempo_init 1
run 2 D_cpi       --corr_scale 3.14159265 --tempo_init 1
run 3 D_cpi_noti  --corr_scale 3.14159265 --tempo_init 0
run 3 D_cpi_tc    --corr_scale 3.14159265 --tempo_init 1 --tempo_corr_scale 0.5
wait
