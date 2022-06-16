dataset=$1
python3 -u -m parser.extract --train_data ${dataset}/camr_train_w_dep.txt.pre
mv *_vocab ${dataset}/

