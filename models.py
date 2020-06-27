#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def GraphAdjacentLayer(d_in, num_dims, jump = 1, use_dropout = False, activation = 'softmax', operator = 'J2'):

  assert activation in ['softmax', 'sigmoid', 'none'];
  x = tf.keras.Input((None, d_in)); # x.shape = (batch, N, d_in)
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

def GConv(d_in, d_out, jump = 1):

  x = tf.keras.Input((None, d_in)); # x.shape = (batch, N, d_in)
  w = tf.keras.Input((None, None, jump)); # w.shape = (batch, N, N, jump)
  w_transposed = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,3,1,2)))(w); # w_transposed.shape = (batch, jump, N, N)
  w_reshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[3])))(w_transposed); # w_reshape.shape = (batch, jump * N, N)
  results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([w_reshape, x]); # results.shape = (batch, jump * N, d_in)
  results = tf.keras.layers.Reshape((jump, -1, d_in))(results); # results.shape = (batch, jump, N, d_in)
  results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,3,1)))(results); # results.shape = (batch, N, d_in, jump)
  results = tf.keras.layers.Reshape((-1, d_in * jump))(results); # results.shape = (batch, N, d_in * jump)
  results = tf.keras.layers.Dense(units = d_out)(results); # results.shape = (batch, N, d_out)
  results = tf.keras.layers.BatchNormalization()(results); # results.shape = (batch, N, d_out)
  return tf.keras.Model(inputs = (x, w), outputs = results);

def GNN(d_in, num_dims, num_layers, num_classes):

  x = tf.keras.Input((None, d_in)); # x.shape = (batch, N, d_in)
  w = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.eye(tf.shape(x)[1]), (1,tf.shape(x)[1],tf.shape(x)[1],1)), (tf.shape(x)[0], 1, 1, 1)))(x); # w.shape = (batch, N, N, 1)
  results = x;
  w_new = w;
  for i in range(num_layers):
    w_new = GraphAdjacentLayer(d_in = d_in + (num_dims // 2) * i, num_dims = num_dims, jump = 2**i, operator = 'J2')([results, w_new]); # w.shape = (batch, N, N, 2)
    results_new = GConv(d_in = d_in + (num_dims // 2) * i, d_out = num_dims // 2, jump = 2**(i+1))([results, w_new]); # x_new.shape = (batch, N, num_dims // 2)
    results_new = tf.keras.layers.LeakyReLU()(results_new); # x_new.shape = (batch, N, num_dims // 2)
    results = tf.keras.layers.Concatenate()([results, results_new]); # x.shape = (batch, N, num_dims * 1.5)
  w_new = GraphAdjacentLayer(d_in = d_in + (num_dims // 2) * num_layers, num_dims = num_dims, jump = 2**num_layers, operator = 'J2')([results, w_new]); # w.shape = (batch, N, N, 2)
  results = GConv(d_in = d_in + (num_dims // 2) * num_layers, d_out = num_classes, jump = 2**(num_layers + 1))([results, w_new]); # x.shape = (batch, N, num_classes)
  return tf.keras.Model(inputs = x, outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  gal = GraphAdjacentLayer(128, 128, jump = 2);
  gal.save('gal.h5');
  a = tf.constant(np.random.normal(size = (8, 10, 128)), dtype = tf.float32);
  w = tf.constant(np.random.normal(size = (8, 10, 10, 2)), dtype = tf.float32);
  b = gal([a,w]);
  print(b.shape);
  gc = GConv(128, 128, jump = 2);
  gc.save('gc.h5');
  b = gc([a,w]);
  print(b.shape);
  gnn = GNN(128, 128, 3, 10);
  gnn.save('gnn.h5');
  b = gnn([a,w]);
  print(b.shape);
