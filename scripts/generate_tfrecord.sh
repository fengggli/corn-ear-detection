image_root=/share/Competition2/images
python generate_tfrecord.py --csv_input=${image_root}/train_labels.csv --image_dir=${image_root} --output_path=${image_root}/train.record
python generate_tfrecord.py --csv_input=${image_root}/test_labels.csv --image_dir=${image_root} --output_path=${image_root}/test.record
