#!/usr/bin/env bash

set -e

# ############### AMR v2.0 ################
# # Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_2.0_utils

# AMR data with **features**
data_dir=./data/silver_data_from_kiro/tamr/
silver_data=$data_dir/bllip.tamr.pred

# ========== Set the above variables correctly ==========

#printf "Cleaning inputs...`date`\n"
#python -u -m stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
#    --amr_files $silver_data
#printf "Done.`date`\n\n"
#
#printf "Recategorizing subgraphs...`date`\n"
#python -u -m stog.data.dataset_readers.amr_parsing.preprocess.recategorizer \
#    --dump_dir ${util_dir} \
#    --amr_files ${silver_data}.input_clean
#printf "Done.`date`\n\n"
#
#printf "Removing senses...`date`\n"
#python -u -m stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
#    --util_dir ${util_dir} \
#    --amr_files ${silver_data}.input_clean.recategorize
#printf "Done.`date`\n\n"
#
#printf "Dependency parsing...`date`\n"
#python -u -m stog.data.dataset_readers.amr_parsing.preprocess.dependency_parsing \
#    --util_dir ${util_dir} \
#    --amr_files ${silver_data}.input_clean.recategorize.nosense
#printf "Done.`date`\n\n"

printf "Dependency parsing...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.filter_valid_amr_graphs \
    --util_dir ${util_dir} \
    --amr_files ${silver_data}
printf "Done.`date`\n\n"

printf "Renaming preprocessed files...`date`\n"
mv ${silver_data}.valid ${silver_data}.preproc
