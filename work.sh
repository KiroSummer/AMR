python3 -u -m parser.work --test_data data/camr2.0_from_junhuili/camr_v2.0/data/origin_camr/camr_origin_train.txt.pre  \
               --test_batch_size 6666 \
               --load_path graph-ckpt/epoch191_batch7999 \
               --beam_size 8\
               --alpha 0.6\
               --max_time_step 100\
               --output_suffix _test_out
