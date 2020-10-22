#!/bin/bash

# postprecessing
for dev_file in ./graph-ckpt/*dev_out.pred; do
  echo "Postprocessing ${dev_file}"
  bash postprocess_2.0.sh ${dev_file}
done

# fast compute smatch
#for dev_post_file in ./ckpt/*.post; do
#  echo "Eval fast smatch ${dev_post_file}"
#  bash ./tools/fast_smatch/compute_smatch.sh ${dev_post_file} ./data/AMR/amr_2.0/dev.txt
#done
