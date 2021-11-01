#python3 train.py --sensitive --imgW 256 --imgH 32 --batch_size 32 --batch_max_length 32 --PAD --lr 2 \
#--train_data lmdb_card/train --valid_data lmdb_card/val \
#--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
#--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111.98.143/best_accuracy.pth \
python3 test.py --sensitive --imgW 256 --imgH 32 --batch_size 512 --batch_max_length 32 --PAD \
--eval_data data/2021_02_18 \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111.98.143/best_accuracy.pth \
--data_filtering_off --workers 16 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
