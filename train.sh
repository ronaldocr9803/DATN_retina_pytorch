export PYTHONPATH=.
python3 train.py \
    --csv_train="dataset_not_aug/pascal_train.csv" \
    --csv_classes="dataset_not_aug/classes.csv" \
    --csv_val="dataset_not_aug/pascal_val.csv" \
    --epochs=50 \
    --depth=101 \
    --weights_folder="./chkpoint_no_aug"