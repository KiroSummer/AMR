#!/usr/bin/env bash

set -e

# Start a Stanford CoreNLP server before running this script.
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

# The compound file is downloaded from
# https://github.com/ChunchuanLv/AMR_AS_GRAPH_PREDICTION/blob/master/data/joints.txt
compound_file=../data/AMR/amr_2.0_utils/joints.txt
silver_amr=/data2/qrxia/data/AMR/silver_data/2m_silver_amr/2m_silver.txt

python -u -m ..stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator \
    $silver_amr \
    --compound_file ${compound_file}
