#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
from models import GNN;
from create_dataset import parse_function;

layers = 3;
class_num = 2;

def main():

  gnn = GNN(7, 128, layers, class_num);
  optimizer = tf.keras.optimizers.Adam(1e-1);
  trainset = tf.data.TFRecordDataset(join('datasets', 'trainset.tfrecord')).repeat(-1).map(parse_function).batch(1).prefetch(tf.data.experimental.AUTOTUNE);
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = gnn, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  log = tf.summary.create_file_writer('checkpoints');
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  for embeddings, weights, labels in trainset:
    # embeddings.shape = (1, N, 7), feature vectors of nodes
    # weights.shape = (1, N, N), adjacent matrix
    # labels.shape = (1, N), class of nodes      
    with tf.GradientTape() as tape:
      features, adjacent = gnn(embeddings); # features.shape = (1, N, class_num), adjacent.shape = (1, N, N, jumps = 16)
      class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(labels, features);
      def body(i, n_jump_adj, loss):
        loss += tf.keras.losses.MSE(n_jump_adj, adjacent[:,:,:,i]);
        i += 1;
        n_jump_adj = tf.linalg.matmul(n_jump_adj, weights); # n_jump_adj.shape = (1, N, N)
        return i, n_jump_adj, loss;
      _, _, edge_loss = tf.while_loop(lambda i, n_jump_adj, loss: i < adjacent.shape[-1], body, loop_vars = (1, weights, 0));
      loss = class_loss + edge_loss;
    avg_loss.update_state(loss);
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(loss, gnn.trainable_variables);
    optimizer.apply_gradients(zip(grads, gnn.trainable_variables));
    if tf.equal(optimizer.iterations % 2000, 0):
      checkpoint.save(join('checkpoints', 'ckpt'));
  if Fasle == exists('model'): mkdir('model');
  gnn.save(join('model', 'gnn.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();