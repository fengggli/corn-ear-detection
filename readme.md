1. codebase description
Code is in https://github.com/fengggli/corn-ear-detection

We use a modified fasterrcnn pipeline from tensorflow object detection repo from https://github.com/tensorflow/models). Two classes are used(cornear connection and corear tail) by us. 

Some important files:
```
tests/
  - detect.py (test script)
  - vis_util.py (visualization util functions)
scripts/
  - faster_rcnn_inception_v2_pet.conf (network and training configuration)
  - generate_tfrecord.py & xml_to_csv.py (data pre-processing)
  - model_main.py (helper function to launch training)

data/examples/ 
  - (example images and video for testing, there is one video and 5 images)

extern/
  - scripts to configure training environment in Linux

notes-train.md
  - instructions for training
```

2. python environment 
```
conda create -n tf-cpu pip pillow matplotlib python=3.7 tensorflow=1.14
conda activate tf-cpu
pip install opencv-python
```

3. test with images
in the tf-cpu conda environment, run the following command in the project root directory:
```
python3 tests/detect.py --imagepath data/examples/file16frame300.jpg
```
The script will print the path of output image in the end.
An example output is in data/predict.jpg, where green boxes are the connection of the cornearn, and yellow boxes are the tail of the cornear.

replace the image path to test with other images.

4. test with example video(can be slow if not having GPU)
Run this command in project root(it will process one video saved in data/example/)
```
python tests/detect.py 
```
