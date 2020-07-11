#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import join, exists;
import json;
import pickle;
import numpy as np;
import cv2;
import tensorflow as tf;

def parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'num': tf.io.FixedLenFeature((), dtype = tf.int64),
      'embeddings': tf.io.VarLenFeature(dtype = tf.float32),
      'labels': tf.io.VarLenFeature(dtype = tf.int64),
      'weights': tf.io.VarLenFeature(dtype = tf.float32)});
  num = tf.cast(feature['num'], dtype = tf.int32);
  embeddings = tf.sparse.to_dense(feature['embeddings'], default_value = 0);
  embeddings = tf.reshape(embeddings, (num, 7));
  labels = tf.sparse.to_dense(feature['labels'], default_value = 0);
  labels = tf.reshape(labels, (num,));
  weights = tf.sparse.to_dense(feature['weights'], default_value = 0);
  weights = tf.reshape(weights, (num, num));
  return embeddings, weights, labels;

def create_dataset(dataset_path):

  if not exists(dataset_path): return False;
  if not exists('datasets'): mkdir('datasets');
  writer = tf.io.TFRecordWriter(join('datasets', 'trainset.tfrecord'));
  with open(join(dataset_path, 'via_project.json'), 'r') as f:
    via = json.load(f);
  data_num = 0;
  classes = dict();
  for fid in via['_via_img_metadata']:
    fn = join(dataset_path, 'images', via['_via_img_metadata'][fid]['filename']);
    img = cv2.imread(fn);
    if img is None:
      print('cannot open image %s' % (fn));
      continue;
    width = img.shape[1];
    height = img.shape[0];
    features = list();
    labels = list();
    for region in via['_via_img_metadata'][fid]['regions']:
      if region['shape_attributes']['name'] != 'rect':
        print('extraction of %s regions not yet implemented!' % region['shape_attributes']['name']);
        continue;
      x = region['shape_attributes']['x'];
      y = region['shape_attributes']['y'];
      w = region['shape_attributes']['width'];
      h = region['shape_attributes']['height'];
      transcript = region['region_attributes']['transcript'];
      region_type = region['region_attributes']['region_type'];
      if region_type not in classes: classes[region_type] = len(classes);
      left = int(max(0, x));
      top = int(max(0, y));
      right = int(min(img.shape[1], x + w));
      bottom = int(min(img.shape[0], y + h));      
      crop = img[top:bottom, left:right, :];
      count = [0,0,0]; # numeric, alphabet, symbol
      for c in transcript:
        if '0' <= c <= '9': count[0] += 1;
        elif 'A' <= c <= 'Z' or 'a' <= c <= 'z': count[1] += 1;
        else: count[2] += 1;
      features.append([left / width, top / height, right / width, bottom / height] + count);
      labels.append(classes[region_type]);
    if len(features) == 0: continue;
    embeddings = np.array(features, dtype = np.float32); # embeddings.shape = (N, 4)
    weights = tf.linalg.band_part(tf.ones((embeddings.shape[0],embeddings.shape[0])), 1, 1) - tf.eye(embeddings.shape[0]); # weights.shape = (N, N)
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'num': tf.train.Feature(int64_list = tf.train.Int64List(value = [len(features)])),
        'embeddings': tf.train.Feature(float_list = tf.train.FloatList(value = embeddings.reshape(-1))),
        'labels': tf.train.Feature(int64_list = tf.train.Int64List(value = labels)),
        'weights': tf.train.Feature(float_list = tf.train.FloatList(value = weights.numpy().reshape(-1)))}));
    writer.write(trainsample.SerializeToString());
    data_num += 1;
  writer.close();
  with open('classes.pkl', 'wb') as f:
    f.write(pickle.dumps(classes));
  print(str(data_num) + " samples were written");
  return True;

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <dataset dir>");
    exit();
  create_dataset(sys.argv[1]);
