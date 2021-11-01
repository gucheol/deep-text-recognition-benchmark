#!/bin/bash
# --select_data hc_aihub_lmdb-hc_gen_lmdb-data_lmdb_release-lmdb_gungsuh --batch_ratio 0.3-0.2-0.3-0.2 \
# 
# --saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth \


# nohup python3 train.py --sensitive --imgW 128 --imgH 32 --batch_size 1024 --batch_max_length 25 --lr 3.0 \
# --PAD \
# --train_data /mnt/b/9fe_Dataset/train --valid_data /mnt/b/9fe_Dataset/val/ \
# --workers 8 \
# --num_iter 200000 --valInterval 1000 \
# --select_data hc_aihub_lmdb-hc_gen_lmdb-data_lmdb_release-lmdb_ui_patch --batch_ratio 0.3-0.1-0.3-0.3 \
# --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn &> out.log &

# nohup python3 train.py --sensitive --imgW 128 --imgH 32 --batch_size 1024 --batch_max_length 25 --lr 3.0 \
# --train_data /mnt/b/9fe_Dataset/train --valid_data /mnt/b/9fe_Dataset/val/ \
# --workers 8 \
# --num_iter 200000 --valInterval 1000 \
# --select_data hc_aihub_lmdb-hc_gen_lmdb-data_lmdb_release-lmdb_ui_patch --batch_ratio 0.3-0.1-0.3-0.3 \
# --Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC &> out.log &

# nohup python3 train.py --sensitive --imgW 128 --imgH 32 --batch_size 1024 --batch_max_length 25 --lr 5.0 \
# --saved_model  saved_models/None-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth  \
# --PAD  \
# --train_data /mnt/b/9fe_Dataset/train --valid_data /mnt/b/9fe_Dataset/val/ \
# --workers 8 \
# --num_iter 100000 --valInterval 1000 \
# --select_data hc_aihub_lmdb-hc_gen_lmdb-data_lmdb_release-lmdb_ui_patch --batch_ratio 0.3-0.1-0.3-0.3 \
# --Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC &> ctc_out.log &

nohup python3 train.py --sensitive --imgW 128 --imgH 32 --batch_size 512 --batch_max_length 25 --lr 2.0 \
--saved_model  saved_models/None-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth  \
--train_data /mnt/b/9fe_Dataset/train --valid_data /mnt/b/9fe_Dataset/val/ \
--workers 8 --PAD \
--num_iter 200000 --valInterval 1000 \
--select_data data_lmdb_release-printed-printed_aug-trdg_train_ui_patch-revised_all_lmdb --batch_ratio 0.2-0.2-0.2-0.2-0.2 \
--Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC &> out.log &
