TODO
=========
1. change checkpoint frequency. ("images" tab can be used for each checkpoint)
2. recheck input labeling
3. try different models.

In sievert
==========

1. ssh -L 2019:localhost:2019 -L 6006:localhost:6006 youname@sievert.cs.iupui.edu (login)
  - 2019 for jupyter port
  - 6006 for tensorboard port

2. in sievert, run
  ```
    source /opt/tf-object-detection/setenv.sh
  ```
3. git clone https://github.com/fengggli/corn-ear-detection.git
4. (in a terminal where you have already sourced  the setenv.sh)jupyter-lab --no-browser --port=2019
5. in your local browser open localhost:2019
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
* r-cnn paper: https://arxiv.org/pdf/1311.2524.pdf
* R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e
* Comparision in speed and accurary of popular methods https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359

Steps
===========
1. Download images with annotations
  ```
  wget  https://cs.iupui.edu/~lifen/files/corndata.tar
  tar -xf ../corndata.tar
  ```
2. cd scripts/ and run 
  ```
  process_data.sh path-to-box-download
  ```
3. change the input_path in scripts/faster_rcnn_inception_xx.config, to tf record path generated in last step

4. train (Use "--model-dir", run with --help for more details)
```shell
  export traindir=/share/Competition2/models/1031-1847 # this variable will be used in both training and generating inference graph
  python model_main.py --logtostderr --model_dir=$traindir --pipeline_config_path=./faster_rcnn_inception_v2_pets.config
```
  Then copy the pipeline file to the traindir, so that we can use it for inference later!

The .config file defines path for tfrecord files, model being used, and training parameters such as optimizer/learning rate, etc.

view tensorboard:
  tensorboard --logdir=$traindir --port

5. export inference graph
  ```shell
  echo $traindir # this will give your the same directory used in last step
  python /opt/tf-object-detection/TensorFlow/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path $traindir/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix $traindir/model.ckpt-14035 --output_directory $traindir/inference_graph
