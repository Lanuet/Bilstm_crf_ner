Vietnamese Ner using character level deep lifelong learning
# Usage: 
# 1. Deep LML
python run.py --train_dir "data/train/name_file_train" --dev_dir "data/dev/name_file_dev" --test_dir "data/test/name_file_test" --lifelong_dir "data/dantri"
# 2: BiLSTM + CRF Model
Switch to branch prefix to run bilstm+crf model with prefix feature

python run.py --train_dir "data/train/name_file_train" --dev_dir "data/dev/name_file_dev" --test_dir "data/test/name_file_test"
