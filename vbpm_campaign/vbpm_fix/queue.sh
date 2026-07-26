#!/bin/bash
# usage: queue.sh <gpu> <jobspec>...   jobspec = "tag|extra args"
cd /home/sogang/jaehoon/VBPM_reintegration
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
L=/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/logs
mkdir -p $L
gpu=$1; shift
for spec in "$@"; do
  tag="${spec%%|*}"; args="${spec#*|}"
  if [[ "$tag" == M_* ]]; then
    common="--mode mert --steps 1200 --warmup 600 --frames 512 --bs 16 --eval_every 400 --n_eval 20"
  else
    common="--mode dirac --steps 800 --warmup 400 --frames 256 --bs 16 --eval_every 400 --n_eval 20"
  fi
  echo "=== $(date +%H:%M:%S) starting $tag on gpu $gpu" >> $L/queue_$gpu.txt
  CUDA_VISIBLE_DEVICES=$gpu $PY vbpm_fix/run_variant_a.py $common --tag $tag $args > $L/$tag.log 2>&1
  echo "=== $(date +%H:%M:%S) done $tag" >> $L/queue_$gpu.txt
done
