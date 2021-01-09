dataset=$1
python3 -u -m silver_data.extract --train_data ${dataset}/train.txt.features.preproc+${dataset}/2m_silver.txt.12w.features.preproc --srl_data no
mv *_vocab ${dataset}/
# python3 encoder.py
# cat ${dataset}/*embed | sort | uniq > ${dataset}/glove.embed.txt
# rm ${dataset}/*embed
