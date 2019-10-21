Data
===========
1. See layerout in data/0-34/feng/images
2. Then run xml-to-csv

Prepare tensorflow model
========================

1. run extern/install_deps.sh, add to PYTHONPATH
2. now can run generate tf-record

Not compatable with tensorflow 1.3:
```
lifen@sievert(:):~/Workspace/dl-assignments/competition2/data/0-34-feng$python /home/lifen/Workspace/dl-assignments/extern/TensorFlow/models/research/object_detection/dataset_tools/create_pascal_tf_record.py --help
Traceback (most recent call last):
  File "/home/lifen/Workspace/dl-assignments/extern/TensorFlow/models/research/object_detection/dataset_tools/create_pascal_tf_record.py", line 38, in <module>
    from object_detection.utils import label_map_util
  File "/home/lifen/Workspace/dl-assignments/extern/TensorFlow/models/research/object_detection/utils/label_map_util.py", line 21, in <module>
    from object_detection.protos import string_int_label_map_pb2
ImportError: cannot import name 'string_int_label_map_pb2' from 'object_detection.protos' (/home/lifen/Workspace/dl-assignments/extern/TensorFlow/models/research/object_detection/protos/__init__.py)

```

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
