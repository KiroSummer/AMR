dataset=$1
python3 -u -m silver_data.extract --train_data ${dataset}/train.txt.features.preproc+${dataset}/bllip.jamr_tamr_spring_0.pred --srl_data no
mv *_vocab ${dataset}/
# python3 encoder.py
# cat ${dataset}/*embed | sort | uniq > ${dataset}/glove.embed.txt
# rm ${dataset}/*embed
