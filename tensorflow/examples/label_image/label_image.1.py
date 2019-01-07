# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import collections
import os.path
import re
import sys
import time

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(sess, file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  # sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def create_image_list(image_dir):
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory " + image_dir + " not found.")
        return None
    
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    
    isRootDir = True
    for sub_dir in sub_dirs:
        if isRootDir:
            isRootDir = False
            continue
        
        extensions = sorted(set(os.path.normcase(ext) for ext in ['JPEG','JPG','jpeg','jpg','PNG','png']))
        file_list = []
        dir_name = os.path.basename(sub_dir)
        
        if dir_name == image_dir:
            continue
        
        tf.logging.info("Looking images in " + dir_name)
        
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
            
        if not file_list:
            tf.logging.warning("No files found")
            continue
        
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        testing_images = []
        
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            testing_images.append(base_name)
        
        result[label_name] = {
            'test' : testing_images,
        }
    
    return result


if __name__ == "__main__":
  startTime = time.time()
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  file_directory = ""
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"
  category_list = {
    'clear': 0,
    'crystals': 1,
    'other': 2,
    'precipitate': 3}

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--image_dir", help="image directory to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:    
    file_name = args.image
  if args.image_dir:
    file_directory = args.image_dir
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  image_list = create_image_list(file_directory)

  true_per_category = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
  }
  total_per_category = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
  }
  data_true = 0
  total_data = 0

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  
  for category in image_list:
    for list in image_list[category]:
      tf.reset_default_graph()
      with tf.Session(graph=graph) as sess:        
        for image in image_list[category][list]:            
          file_name = os.path.join(file_directory, category, image)
          t = read_tensor_from_image_file(
            sess,
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

          results = sess.run(output_operation.outputs[0], {
                      input_operation.outputs[0]: t
                  })
          results = np.squeeze(results)
          result = results.argsort()[::-1]
          total_data += 1
          total_per_category[category_list[category]] += 1
          if result[0] == category_list[category]:
              data_true += 1
              true_per_category[category_list[category]] +=1

          accuracy = (data_true / total_data) * 100
          sys.stdout.write('\r>> Processing %d images. Total data true = %d. Accuracy %.2f%% \n' % (total_data, data_true, accuracy))
          sys.stdout.flush()
  
  sys.stdout.write('Clear accuracy %.2f%% \r\n' % ((true_per_category[0]/total_per_category[0]) * 100))
  sys.stdout.flush()
  sys.stdout.write('Crystals accuracy %.2f%% \r\n' % ((true_per_category[1]/total_per_category[1]) * 100))
  sys.stdout.flush()
  sys.stdout.write('Other accuracy %.2f%% \r\n' % ((true_per_category[2]/total_per_category[2]) * 100))
  sys.stdout.flush()
  sys.stdout.write('Precipitate accuracy %.2f%% \r\n' % ((true_per_category[3]/total_per_category[3]) * 100))
  sys.stdout.flush()
  endTime = time.time()
  runningTime = endTime - startTime
  sys.stdout.write('Process done in %d s! Final accuracy in average %.2f%% \r\n' % (runningTime, accuracy))
  sys.stdout.flush()