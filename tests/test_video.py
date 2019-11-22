import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

'''
tf_extern_root = '/home/lifen/Workspace/corn-data/extern/TensorFlow/models/research/'
sys.path.append(tf_extern_root)
sys.path.append(tf_extern_root + 'slim')
sys.path.append(tf_extern_root + 'object_detection')
'''

sys.path.append('./scripts')
import vis_util
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--imagepath", help="image to process(if not provided, a sample video will be processed instead)")

args = parser.parse_args()

# operato on a video if not providing any image input
input_is_video=True
videopath='data/examples/GOPR0388.MP4'

# check file exits
if(args.imagepath):
    imagepath=args.imagepath
    outputpath='data/predict.jpg'
    input_is_video=False


MODEL_PATH='data/saved_models/v0.1/'

PATH_TO_CKPT = os.path.join(MODEL_PATH, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(MODEL_PATH, 'corn_label_map_2class.pbtxt')
NUM_CLASSES = 2

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
    # label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
category_index = {1:{'id':1, 'name': 'cornear'}, 2:{'id':2, 'name': 'rearside'}}

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

if(input_is_video):
    print("This is a video!")
    cap = cv2.VideoCapture(videopath)  # Change only if you have more than one webcams
else:
    try:
        this_image = Image.open(imagepath)
    except IOError:
        print("image path", imagepath,"is invalid")

    image_np = load_image_into_numpy_array(this_image)

# Detection a video
count = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            if input_is_video:
                print("frame #", count)
                ret, image_np = cap.read()
                if ret:
                    count += 5 # i.e. at 30 fps, this advances one second
                    cap.set(1, count)
                else:
                    cap.release()
                    break
            else:
                print("processing image at ", imagepath)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display output
            if input_is_video:
                cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                plt.figure(figsize=(800, 600))
                plt.imshow(image_np)
                plt.imsave(outputpath, image_np)
                print('predicted image is saved in', outputpath)
                break
