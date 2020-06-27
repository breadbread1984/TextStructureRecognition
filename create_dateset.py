#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import join, exists;
import json;
import numpy as np;
import cv2;
import tensorflow as tf;

def parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'num': tf.io.FixedLenFeature((), dtype = tf.int64),
      'embeddings': tf.io.VarLenFeature(dtype = tf.float32),
      'weights': tf.io.VarLenFeature(dtype = tf.float32)});
  num = tf.cast(feature['num'], dtype = tf.int32);
  embeddings = tf.sparse.to_dense(feature['embeddings'], default_value = 0);
  embeddings = tf.reshape(embeddings, (num, 4));
  weights = tf.sparse.to_dense(feature['weights'], default_value = 0);
  weights = tf.reshape(weights, (num, num));
  return embeddings, weights;

def create_dataset(dataset_path):

  if not exists(dataset_path): return False;
  if not exists('datasets'): mkdir('datasets');
  writer = tf.io.TFRecordWriter(join('datasets', 'trainset.tfrecord'));
  with open(join(dataset_path, 'via_project.json'), 'r') as f:
    via = json.load(f);
  count = 0;
  for fid in via['_via_img_metadata']:
    fn = join(dataset_path, 'images', via['_via_img_metadata'][fid]['filename']);
    img = cv2.imread(fn);
    if img is None:
      print('cannot open image %s' % (fn));
      continue;
    width = img.shape[1];
    height = img.shape[0];
    positions = list();
    for region in via['_via_img_metadata'][fid]['regions']:
      if region['shape_attributes']['name'] != 'rect':
        print('extraction of %s regions not yet implemented!' % region['shape_attributes']['name']);
        continue;
      x = region['shape_attributes']['x'];
      y = region['shape_attributes']['y'];
      w = region['shape_attributes']['width'];
      h = region['shape_attributes']['height'];
      left = int(max(0, x));
      top = int(max(0, y));
      right = int(min(img.shape[1], x + w));
      bottom = int(min(img.shape[0], y + h));      
      crop = img[top:bottom, left:right, :];
      positions.append([left / width, top / height, right / width, bottom / height]);
    if len(positions) == 0: continue;
    embeddings = np.array(positions, dtype = np.float32); # embeddings.shape = (N, 4)
    weights = tf.linalg.band_part(tf.ones((embeddings.shape[0],embeddings.shape[0])), 1, 1) - tf.eye(embeddings.shape[0]); # weights.shape = (N, N)
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'num': tf.train.Feature(int64_list = tf.train.Int64List(value = [len(positions)])),
        'embedings': tf.train.Feature(float_list = tf.train.FloatList(value = embeddings.reshape(-1))),
        'weights': tf.train.Feature(float_list = tf.train.FloatList(value = weights.numpy().reshape(-1)))}));
    writer.write(trainsample.SerializeToString());
    count += 1;
  writer.close();
  print(str(count) + " samples were written");
  return True;

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <dataset dir>");
    exit();
  create_dataset(sys.argv[1]);
