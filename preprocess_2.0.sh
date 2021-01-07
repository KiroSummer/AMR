#!/usr/bin/env bash

set -e

# ############### AMR v2.0 ################
# # Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_2.0_utils

# AMR data with **features**
data_dir=data/AMR/amr_2.0_with_dependency_arc_and_rel
train_data=${data_dir}/train.txt.features
dev_data=${data_dir}/dev.txt.features
test_data=${data_dir}/test.txt.features

# ========== Set the above variables correctly ==========

printf "Cleaning inputs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
    --amr_files ${train_data} ${dev_data} ${test_data}
printf "Done.`date`\n\n"

printf "Recategorizing subgraphs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.recategorizer \
    --dump_dir ${util_dir} \
    --amr_files ${train_data}.input_clean ${dev_data}.input_clean
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.text_anonymizor \
    --amr_file ${test_data}.input_clean \
    --util_dir ${util_dir}
printf "Done.`date`\n\n"

printf "Removing senses...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.sense_remover \
    --util_dir ${util_dir} \
    --amr_files ${train_data}.input_clean.recategorize \
    ${dev_data}.input_clean.recategorize \
    ${test_data}.input_clean.recategorize
printf "Done.`date`\n\n"

printf "Dependency parsing...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.dependency_parsing \
    --util_dir ${util_dir} \
    --amr_files ${train_data}.input_clean.recategorize.nosense \
    ${dev_data}.input_clean.recategorize.nosense \
    ${test_data}.input_clean.recategorize.nosense
printf "Done.`date`\n\n"

printf "Renaming preprocessed files...`date`\n"
mv ${test_data}.input_clean.recategorize.nosense.dep ${test_data}.preproc
mv ${train_data}.input_clean.recategorize.nosense.dep ${train_data}.preproc
mv ${dev_data}.input_clean.recategorize.nosense.dep ${dev_data}.preproc
rm ${data_dir}/*.input_clean*
