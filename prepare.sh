dataset=$1
python3 -u -m parser.extract --train_data ${dataset}/train.txt.features.preproc
mv *_vocab ${dataset}/

