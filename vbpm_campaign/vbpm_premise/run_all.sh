#!/bin/sh
P=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
cd /home/sogang/jaehoon/VBPM_reintegration/vbpm_premise
for f in p0_sanity.py p1a_monotonicity.py p1b_constant_advance.py p1c_beatlevel.py \
         p1c_transition.py p1d_tempo.py p1d2_decompose.py p1d3_shape.py p1d4_lawselect.py; do
  echo "################################################## $f"
  $P $f 2>&1 | grep -v Warning
done
