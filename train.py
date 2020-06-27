#!/usr/bin/python3

import tensorflow as tf;
from models import GNN;
from create_dataset import parse_function;

def main():

  trainset = tf.data.TFRecordDataset(join('datasets', 'trainset.tfrecord')).repeat(-1).map(parse_function).batch(1).prefetch(tf.data.experimental.AUTOTUNE);
  for embeddings, weights in trainset:
    print(embeddings, weights);
    break;

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
