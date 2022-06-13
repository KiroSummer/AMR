dataset=$1
python3 -u -m parser.train --info hard-mtl-loss-weights\
                --tok_vocab ${dataset}/tok_vocab\
                --lem_vocab ${dataset}/lem_vocab\
                --pos_vocab  ${dataset}/pos_vocab\
                --ner_vocab ${dataset}/ner_vocab\
                --dep_rel_vocab ${dataset}/dep_rel_vocab\
                --srl_vocab ${dataset}/srl_vocab\
                --concept_vocab ${dataset}/concept_vocab\
                --predictable_concept_vocab ${dataset}/predictable_concept_vocab\
                --rel_vocab ${dataset}/rel_vocab\
                --word_char_vocab ${dataset}/word_char_vocab\
                --concept_char_vocab ${dataset}/concept_char_vocab\
                --train_data ${dataset}/debug_train.txt.pre \
                --dev_data ${dataset}/debug_train.txt.pre \
                --srl_data ./data/auto-srl/train.english.conll05.jsonlines.features \
                --with_bert \
                --bert_path hfl/chinese-roberta-wwm-ext \
                --word_char_dim 32\
                --word_dim 300\
                --pos_dim 0\
                --ner_dim 0\
                --dep_rel_dim 0\
                --concept_char_dim 32\
                --concept_dim 300 \
                --rel_dim 100 \
                --cnn_filter 3 256\
                --char2word_dim 128\
                --char2concept_dim 128\
                --embed_dim 512\
                --ff_embed_dim 1024\
                --num_heads 8\
                --pred_size 200\
                --argu_size 200\
                --span_size 200\
                --ffnn_size 200\
                --ffnn_depth 1\
                --snt_layers 4\
                --graph_layers 2\
                --inference_layers 4\
                --dropout 0.2\
                --unk_rate 0.33\
                --epochs 2000\
                --train_batch_size 4444\
                --dev_batch_size 4444 \
                --lr_scale 1. \
                --weight_decay 1e-7 \
                --warmup_steps 1000\
                --print_every 1 \
                --eval_every 1 \
                --batches_per_update 4 \
                --ckpt ckpt\
                --world_size 2\
                --gpus 2\
                --MASTER_ADDR localhost\
                --MASTER_PORT 29545\
                --start_rank 0
