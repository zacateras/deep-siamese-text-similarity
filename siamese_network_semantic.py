import tensorflow as tf
import numpy as np

class SiameseLSTMw2v(object):
  """
  A LSTM based deep Siamese network for text similarity.
  Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
  """
  
  def stackedRNN(self, x, dropout, scope, embedding_size, out_units, hidden_units, layers):
    # Prepare data shape to match `static_rnn` function requirements
    x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    # print(x)
    # Define lstm cells with tensorflow
    # Forward direction cell

    with tf.name_scope("fw"+scope), tf.variable_scope("fw"+scope):
      stacked_rnn_fw = []
      for i in range(layers):
        units = out_units if i == (layers-1) else hidden_units
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(units, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
        stacked_rnn_fw.append(lstm_fw_cell)
      lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

      outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
    return outputs[-1]

  def contrastive_loss(self, y, d, batch_size):
    tmp = y * tf.square(d)
    #tmp= tf.mul(y,tf.square(d))
    tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
    return tf.reduce_sum(tmp + tmp2) / batch_size / 2
  
  def __init__(
    self, sequence_length, vocab_size, embedding_size, batch_size, trainableEmbeddings, sides_out_units,
    side1_layers        , side2_layers,
    side1_hidden_units  , side2_hidden_units):

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
      self.out1 = self.stackedRNN(self.embedded_words1, self.side1_dropout, "side1", embedding_size, sides_out_units, side1_hidden_units, side1_layers)
      self.out2 = self.stackedRNN(self.embedded_words2, self.side2_dropout, "side2", embedding_size, sides_out_units, side2_hidden_units, side2_layers)

      # out1 and out2 are lists of vectors outputted by stacked LSTMs for each input pair of sentences (i)
      # so here we compute distance: norm2(out1(i) - out2(i)) / (norm2(out1(i)) + norm2(out2(i)))
      # this can result in outputting a number between 0 and 1
      self.distance = tf.div(
        tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True)),
        tf.add(
          tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))

      self.distance = tf.reshape(self.distance, [-1], name="distance")
      self.output_y_norm = tf.subtract(tf.ones_like(self.distance), self.distance)

    with tf.name_scope("loss"):
      self.loss = self.contrastive_loss(self.input_y_norm, self.distance, batch_size)

    with tf.name_scope("accuracy_gs"):
      self.pred_gs = tf.rint(tf.scalar_mul(5.0, self.output_y_norm), name="pred_gs")
      y_gs = tf.rint(tf.scalar_mul(5.0, self.input_y_norm))
      correct_predictions = tf.equal(self.pred_gs, y_gs)
      self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy_gs")

    with tf.name_scope("pcc"):
      input_y_norm_minus_mean = tf.subtract(self.input_y_norm, tf.reduce_mean(self.input_y_norm))
      output_y_norm_minus_mean = tf.subtract(self.output_y_norm, tf.reduce_mean(self.output_y_norm))

      numerator = tf.reduce_sum(tf.multiply(input_y_norm_minus_mean, output_y_norm_minus_mean))
      denominator = tf.multiply(
        tf.sqrt(tf.reduce_sum(tf.square(input_y_norm_minus_mean))),
        tf.sqrt(tf.reduce_sum(tf.square(output_y_norm_minus_mean))))

      self.pcc = tf.divide(numerator, denominator)

    with tf.name_scope("rho"):
      pass

    with tf.name_scope("mse"):
      # SUM_i((y_i-(1-d_i))^2)) i from 1 to n
      self.mse = tf.reduce_sum(tf.square(tf.subtract(self.input_y_norm, self.output_y_norm)))
