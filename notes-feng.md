In sievert
==========

1. ssh -L 2019:localhost:2019 youname@sievert.cs.iupui.edu (login)
2. in sievert, run
  ```
    source /opt/tf-object-detection/setenv.sh
  ```
3. git clone https://github.com/fengggli/corn-ear-detection.git
4. jupyter-lab --no-brower --port=2019
5. in your local brower open localhost:2019
6. data is /share

Deps
=========
Update: see https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
1. cd extern
1. ./install_deps.sh
2. source setenv.sh
3. run ``python $TFMODELSPATH/object_detection/builders/model_builder_test.py`` to test!(make sure using correct conda env!)

Read the installation guide carefully:
  https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/installation.md
conda install lxml pillow

COCO
=========
* ModuleNotFoundError: No module named 'pycocotools'
* See https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#coco-api-installation-optional
* need conda install cython
* research$ln -s ./cocoapi/PythonAPI/pycocotools/ .


Model
===========

* ref: https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/detection_model_zoo.md
* model downloaded in scripts, from http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Steps
===========
1. See file layerout in data/0-34/feng/images

2. Then run xml_to_csv.py 
  ```
  cd data/0-34-feng
  python ../../scripts/xml_to_csv.py
  ```

3. generate tfrecords(in scripts dir):
  (run scripts/generate_tf_record.sh)
  ```
  cd scripts
  python generate_tfrecord.py --csv_input=../data/0-34-feng/images/train_labels.csv --image_dir=../data/images_original/0-34-feng --output_path=../data/0-34-feng/train.record
  python generate_tfrecord.py --csv_input=../data/0-34-feng/images/test_labels.csv --image_dir=../data/images_original/0-34-feng --output_path=../data/0-34-feng/test.record
  ```

4. train
```shell
  python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=./faster_rcnn_inception_v2_pets.config
  #somehow the data is not stored in training. Instead it's in  tmp directory (observed from log)
  mkdir train_copy && cp -r /tmp/tmpbjnn4igd/ training_copy
```

The .config file defines path for tfrecord files, model being used, and training parameters such as optimizer/learning rate, etc.

5. export inference graph 
  ```
  python /home/lifen/Workspace/corn-ear-detection/extern/TensorFlow/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training_copy/model.ckpt-2000 --output_directory inference_graph
  ```

6. run the detect.ipynb to predict (Work in progress).

