#!/usr/bin/env bash

set -e

# ############### CAMR v2.0 ################
# AMR data
data_dir=../data/camr2.0_from_junhuili/camr_v2.0/data/origin_camr
train_data=${data_dir}/camr_origin_train.txt
dev_data=${data_dir}/camr_origin_dev.txt
test_data=${data_dir}/camr_origin_test.txt

# ========== Set the above variables correctly ==========
printf "Converting CAMR format to AMR format...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.camr_graph_to_amr \
    --amr_files ${train_data} ${dev_data} ${test_data}
printf "Done.`date`\n\n"
