#!/usr/bin/env bash

set -e

# ############### AMR v2.0 ################
# # Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_2.0_utils

# AMR data with **features**
data_dir=/data2/qrxia/AMR-research/kamr-gr-2.0/data/silver_data_from_kiro/spring/
silver_data=$data_dir/bllip.ber5.sents.txt.spring.features 

# ========== Set the above variables correctly ==========

printf "Cleaning inputs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
    --amr_files $silver_data
printf "Done.`date`\n\n"

printf "Recategorizing subgraphs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.recategorizer \
    --dump_dir ${util_dir} \
    --amr_files ${silver_data}.input_clean
printf "Done.`date`\n\n"

printf "Removing senses...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
    --util_dir ${util_dir} \
    --amr_files ${silver_data}.input_clean.recategorize
printf "Done.`date`\n\n"

printf "Dependency parsing...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.dependency_parsing \
    --util_dir ${util_dir} \
    --amr_files ${silver_data}.input_clean.recategorize.nosense \
printf "Done.`date`\n\n"

printf "Renaming preprocessed files...`date`\n"
mv ${silver_data}.input_clean.recategorize.nosense.dep ${silver_data}.preproc
rm ${data_dir}/*.input_clean*
