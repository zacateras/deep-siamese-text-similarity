import tensorflow as tf
import numpy as np

class SiameseLSTMw2v(object):
  """
  A LSTM based deep Siamese network for text similarity.
  Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
  """
  
  def stackedRNN(self, x, dropout, scope, embedding_size, nodes):
    # Prepare data shape to match `static_rnn` function requirements
    x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

    with tf.name_scope(scope), tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      stacked_rnn = []
      
      for n in nodes:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(int(n), name=('lstm_cell_%s' % n))
        lstm_cell_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)
        stacked_rnn.append(lstm_cell_dropout)

      lstm_cell_m = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn)
      outputs, _ = tf.nn.static_rnn(lstm_cell_m, x, dtype=tf.float32)
    return outputs[-1]

  def contrastive_loss(self, y, d, batch_size):
    tmp = y * tf.square(d)
    # tmp= tf.mul(y,tf.square(d))
    tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
    return tf.reduce_sum(tmp + tmp2) / batch_size / 2

  def calculate_pcc(self, x, y, name=None):
    x_minus_mean = tf.subtract(x, tf.reduce_mean(x))
    y_minus_mean = tf.subtract(y, tf.reduce_mean(y))

    numerator = tf.reduce_sum(tf.multiply(x_minus_mean, y_minus_mean))
    denominator = tf.multiply(
      tf.sqrt(tf.reduce_sum(tf.square(x_minus_mean))),
      tf.sqrt(tf.reduce_sum(tf.square(y_minus_mean))))

    return tf.divide(numerator, denominator, name=name)

  def calculate_rho(self, x, y, batch_size, name=None):
    x_rank = tf.add(tf.contrib.framework.argsort(x), 1)
    y_rank = tf.add(tf.contrib.framework.argsort(y), 1)
    d_rank = tf.subtract(x_rank, y_rank)

    numerator = tf.cast(tf.multiply(6, tf.reduce_sum(tf.square(d_rank))), tf.float32)
    denominator = batch_size * (batch_size * batch_size - 1.0)
    
    return tf.subtract(1.0, tf.divide(numerator, denominator), name=name)

  def calculate_mse(self, x, y, name=None):
    return tf.reduce_mean(tf.abs(tf.subtract(x, y)), name=name)
  
  def __init__(
    self, sequence_length, vocab_size, embedding_size, batch_size, trainableEmbeddings, tied, side1_nodes, side2_nodes):
    
    # Placeholders for input, output
    self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
    self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
    self.input_y_norm = tf.placeholder(tf.float32, [None], name="input_y_norm")

    # Placeholders for parameters - dropout, layers, hidden_units
    self.side1_dropout = tf.placeholder(tf.float32, name="side1_dropout")
    self.side2_dropout = tf.placeholder(tf.float32, name="side2_dropout")

    # Embedding layer
    with tf.name_scope("embedding"):
      self.W = tf.Variable(
        tf.constant(0.0, shape=[vocab_size, embedding_size]),
        trainable=trainableEmbeddings, name="W")
      self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
      self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)

    # Create a convolution + maxpool layer for each filter size
    with tf.name_scope("output"):
      self.out1 = self.stackedRNN(self.embedded_words1, self.side1_dropout, "shared" if tied else "side_1", embedding_size, side1_nodes)
      self.out2 = self.stackedRNN(self.embedded_words2, self.side2_dropout, "shared" if tied else "side_2", embedding_size, side2_nodes)

      # out1 and out2 are lists of vectors outputted by stacked LSTMs for each input pair of sentences (i)
      # so here we compute distance: norm2(out1(i) - out2(i)) / (norm2(out1(i)) + norm2(out2(i)))
      # this can result in outputting a number between 0 and 1
      #
      # NOTE: out1 - out2 can often be 0, the framework have troubles with computing gradients for 0 = sqrt(0) (identity)
      #       thus 1e-16 is added here as minimal total distance between out1 and out2
      self.distance = tf.div(
        tf.sqrt(tf.add(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keepdims=True), 1e-16)),
        tf.add(
          tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keepdims=True)),
          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keepdims=True))))

      self.distance = tf.reshape(self.distance, [-1], name="distance")
      self.output_y_norm = tf.subtract(tf.ones_like(self.distance), self.distance)

    with tf.name_scope("loss"):
      self.loss = self.contrastive_loss(self.input_y_norm, self.distance, batch_size)

    # Gold standard accuracy will not be used
    # with tf.name_scope("accuracy_gs"):
    #   pred_gs = tf.rint(tf.scalar_mul(5.0, self.output_y_norm))
    #   y_gs = tf.rint(tf.scalar_mul(5.0, self.input_y_norm))
    #   correct_predictions = tf.equal(pred_gs, y_gs)
    #   self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy_gs")

    with tf.name_scope("pcc"):
      self.pcc = self.calculate_pcc(self.input_y_norm, self.output_y_norm, name="pcc")

    with tf.name_scope("rho"):
      self.rho = self.calculate_rho(self.input_y_norm, self.output_y_norm, batch_size, name="rho")

    with tf.name_scope("mse"):
      self.mse = self.calculate_mse(self.input_y_norm, self.output_y_norm, name="mse")
