
#Aufbauend auf dem Object Detection Tutorial von TF

# -----------------------------------------------------> Imports <-----------------------------------------------------

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2
import pyrealsense2 as rs


# ---------------------------------------------------> Camera Setup <---------------------------------------------------


pipeline = rs.pipeline()
config = rs.config()

# Stream Einstellungen

config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30) # Welches Format eignet sich am besten?
config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8,30)

# cap = cv2.VideoCapture(0)  # Captured


# ------------------------------------------------> Model preparation <-------------------------------------------------


# WICHTIG!!
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing
# `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.
# Andere Out of the Box Models:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Download Model

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# -------------------------------------------------> Loading label map <------------------------------------------------

''' 
Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  
Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
'''

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# -----------------------------------------------------> Detection <----------------------------------------------------


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:

        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)

        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

  return output_dict


# ------------------------------------------> Starten des Intel Realsense Streams <-------------------------------------

profile = pipeline.start(config)

try:
    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        # pixel_distance_in_meters = depth_frame.get_distance(x, y)
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # ---------------------------------------- Filtering the Depth Stream ------------------------------------------



        # ---------------------------------------> Visualization ------------------------------------------------------

        # Konvertieren der Bilddaten in numpy Arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Alternative Darstellung des Depth Streams
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03),cv2.COLORMAP_JET)

        # hier depth_colormap einsetzen für alternative Darstellung des Depth streams
        rs_image = np.hstack((color_image, colorized_depth))  # Zeigt die beiden Streams nebeneinander in einem Fenster an

        #rs_image = color_image + colorized_depth  # Zeigt die beiden Streams übereinander an


        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(rs_image, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(rs_image, detection_graph)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            rs_image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)


        # ------------------------------------------> Distance to Object -----------------------------------------------
        ''' Experimental: Keine Ahnung wie coordinaten aus dem Boxes bekommt

        for i,b in enumerate (output_dict['detection_boxes']):  # Boxes contain the coordinates
            mid_x = (output_dict['detection_boxes'][i][3] + output_dict['detection_boxes'][i][1]) / 2
            mid_y = (output_dict['detection_boxes'][i][2] + output_dict['detection_boxes'][i][0]) / 2
            apx_distance = round((1-(output_dict['detection_boxes'][i][3] - output_dict['detection_boxes'][i][1]))**4, 1)

            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*640), int(mid_y*480)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        '''


        # Ausgabe

        cv2.namedWindow('3D Object Detection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('3D Object Detection', depth_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Exit mit 'q'
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()