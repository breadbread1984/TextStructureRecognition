#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def GraphAdjacentLayer(num_features, num_dims, jump = 1, use_dropout = False, activation = 'softmax', operator = 'J2'):

  assert activation in ['softmax', 'sigmoid', 'none'];
  x = tf.keras.Input((None, num_features)); # x.shape = (batch, N, d_in)
  w = tf.keras.Input((None, None, jump)); # w.shape = (batch, N, N, jump)
  xi = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 2))(x); # xi.shape = (batch, N, 1, d_in)
  xj = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(x); # xj.shape = (batch, 1, N, d_im)
  results = tf.keras.layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))([xi, xj]); # results.shape = (batch, N, N, d_in)
  results = tf.keras.layers.Dense(units = 2 * num_dims)(results); # results.shape = (batch, N, N, num_dims * 2)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  if use_dropout: results = tf.keras.layers.Dropout()(results);
  results = tf.keras.layers.Dense(units = 2 * num_dims)(results); # results.shape = (batch, N, N, num_dims * 2)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Dense(units = num_dims)(results); # results.shape = (batch, N, N, num_dims)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Dense(units = num_dims)(results); # results.shape = (batch, N, N, num_dims)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Dense(units = jump)(results); # results.shape = (batch, N, N, jump)
  if activation == 'softmax':
    results = tf.keras.layers.Lambda(lambda x: x[0] - x[1] * 1e8)([results, w]); # results.shape = (batch, N, N, jump)
    flatten = tf.keras.layers.Reshape((-1, jump))(results); # flatten.shape = (batch, N * N, jump)
    weights = tf.keras.layers.Softmax(axis = -2)(flatten); # weights.shape = (batch, N * N, jump)
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], tf.shape(x[1])))([weights, results]); # results.shape = (batch, N, N, jump)
  elif activation == 'sigmoid':
    results = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x))(results); # results.shape = (batch, N, N, jump)
    results = tf.keras.layers.Lambda(lambda x: x[0] * (1 - x[1]))([results, w]); # results.shape = (batch, N, N, jump)
  elif activation == 'none':
    results = tf.keras.layers.Lambda(lambda x: x[0] * (1 - x[1]))([results, w]); # results.shape = (batch, N, N, jump)
  else:
    raise Exception('unknown activation!');
  if operator == 'laplace':
    results = tf.keras.layers.Lambda(lambda x: x[1] - x[0])([results, w]); # results.shape = (batch, N, N, jump)
  elif operator == 'J2':
    results = tf.keras.layers.Concatenate(axis = -1)([results, w]); # results.shape = (batch, N, N, 2 * jump)
  else:
    raise Exception('unknown operator!');
  return tf.keras.Model(inputs = (x, w), outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  gal = GraphAdjacentLayer(128, 128, jump = 2);
  gal.save('gal.h5');
  a = tf.constant(np.random.normal(size = (8, 10, 128)), dtype = tf.float32);
  w = tf.constant(np.random.normal(size = (8, 10, 10, 2)), dtype = tf.float32);
  b = gal([a,w]);
  print(b.shape);
