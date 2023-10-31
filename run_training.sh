PATH_DATASET_DIR=''
python main.py --name_model=simple_classifier --path_dataset=$PATH_DATASET_DIR --type_training=custom_training --epochs=20 --learning_rate=0.00001 --backbone=resnet101
