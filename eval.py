#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
from tensorflow.contrib import learn
from input_helpers import InputHelper

# Parameters
# ==================================================
tf.flags.DEFINE_string("eval_filepath", None, "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("log_filepath", None, "Result log file path (Default: None)")
tf.flags.DEFINE_string("log_event", None, "Result log event name (Default: None)")
tf.flags.DEFINE_float("y_scale", 5.0, "scale of y in evaluation file (default: 5.0)")
tf.flags.DEFINE_integer("y_position", 0, "position of y in evaluation file (default: 0)")
tf.flags.DEFINE_integer("x1_position", 1, "position of x1 in training file (default: 1)")
tf.flags.DEFINE_integer("x2_position", 2, "position of x2 in training file (default: 2)")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1526593435/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1512222837/checkpoints/model-5000", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

print("EXECUTION PARAMETERS:")
for attr, flag in sorted(FLAGS.__flags.items()):
  print("{}={}".format(attr.upper(), flag.value))

if FLAGS.eval_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None:
  print("Eval or Vocab filepaths are empty.")
  exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test, x2_test, y_test = inpH.getTestDataSet(
  FLAGS.eval_filepath, FLAGS.y_position, FLAGS.x1_position, FLAGS.x2_position, FLAGS.vocab_filepath, 30)

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
graph = tf.Graph()
with graph.as_default():
  session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
  sess = tf.Session(config=session_conf)
  with sess.as_default():
    # Load the saved meta graph and restore variables
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
    input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
    input_y_norm = graph.get_operation_by_name("input_y_norm").outputs[0]

    side1_dropout = graph.get_operation_by_name("side1_dropout").outputs[0]
    side2_dropout = graph.get_operation_by_name("side2_dropout").outputs[0]

    # Tensors we want to evaluate
    predictions = graph.get_operation_by_name("output/distance").outputs[0]
    pcc = graph.get_operation_by_name("pcc/pcc").outputs[0]
    rho = graph.get_operation_by_name("rho/rho").outputs[0]
    mse = graph.get_operation_by_name("mse/mse").outputs[0]

    # Generate batches for one epoch
    batches = inpH.batch_iter(list(zip(x1_test, x2_test, y_test)), 2 * FLAGS.batch_size, 1, shuffle=False)

    # Collect the predictions here
    all_predictions = []
    all_pcc = []
    all_rho = []
    all_mse = []

    for db in batches:
      x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
      batch_predictions, batch_pcc, batch_rho, batch_mse = sess.run([
          predictions,
          pcc,
          rho,
          mse
        ], { 
          input_x1: x1_dev_b,
          input_x2: x2_dev_b,
          input_y_norm: map(lambda x: x / FLAGS.y_scale, y_dev_b),
          side1_dropout: 1.0,
          side2_dropout: 1.0
        })
      
      # rescale mse to fit initial domain
      batch_mse = batch_mse * FLAGS.y_scale

      all_predictions = np.concatenate([all_predictions, batch_predictions])
      all_pcc = np.concatenate([all_pcc, [batch_pcc]])
      all_rho = np.concatenate([all_rho, [batch_rho]])
      all_mse = np.concatenate([all_mse, [batch_mse]])

    pcc = np.mean(all_pcc)
    rho = np.mean(all_rho)
    mse = np.mean(all_mse)

    print("AVG PCC:{:g} RHO:{:g} MSE:{:g}".format(pcc, rho, mse))

    # write evaluation log tsv event entry
    if FLAGS.log_filepath is not None and FLAGS.log_event is not None:
      with open(FLAGS.log_filepath, "a+") as f:
        f.write('%s\t%s\t%s\t%s\r\n' % (FLAGS.log_event, pcc, rho, mse))

