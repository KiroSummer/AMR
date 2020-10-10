#!/bin/bash

# Test
#for dev_file in ./kiro-ckpt/epoch*_batch*9; do
#  echo "Test on checkpoint ${dev_file}"
#  bash test.sh ${dev_file} $1
#done

# postprocessing_2.0
for test_file in ./kiro-ckpt/epoch*_test*.pred; do
  echo "Postprocessing2.0 ${test_file}"
  bash postprocessing_2.0.sh ${test_file}
done

# fast compute smatch
#for dev_post_file in ./ckpt/*.post; do
#  echo "Eval fast smatch ${dev_post_file}"
#  bash ./tools/fast_smatch/compute_smatch.sh ${dev_post_file} ./data/AMR/amr_2.0/dev.txt
#done
