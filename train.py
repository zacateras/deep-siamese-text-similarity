#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
import sys
import shutil
from input_helpers import InputHelper
from siamese_network_semantic import SiameseLSTMw2v
from tensorflow.contrib import learn
import gzip
from random import random

# Parameters
# ==================================================
tf.flags.DEFINE_string("training_filepath", "data/train_snli.txt", "training file path (default: None)")
tf.flags.DEFINE_string("output_dirpath", None, "output directory path (default: None)")
tf.flags.DEFINE_float("y_scale", 5.0, "scale of y in training file (default: 5.0)")
tf.flags.DEFINE_integer("y_position", 0, "position of y in training file (default: 0)")
tf.flags.DEFINE_integer("x1_position", 0, "position of x1 in training file (default: 1)")
tf.flags.DEFINE_integer("x2_position", 0, "position of x2 in training file (default: 2)")

# Embedding parameters
tf.flags.DEFINE_string("word2vec_model", "wiki.simple.vec", "word2vec pre-trained embeddings file (default: None)")
tf.flags.DEFINE_string("word2vec_format", "text", "word2vec pre-trained embeddings file format (bin/text/textgz)(default: None)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")

# RNN stack parameters
tf.flags.DEFINE_boolean("tied", True, "Different side weights are tied / untied (default: True)")
tf.flags.DEFINE_float("side1_dropout", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("side2_dropout", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_list("side1_nodes", [50, 50, 50], "Number of nodes in layers for Side_1 (default:50,50,50)")
tf.flags.DEFINE_list("side2_nodes", [50, 50, 50], "Number of nodes in layers for Side_2 (default:50,50,50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("max_iterations", 500000, "Maximum number of iterations")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

print("EXECUTION PARAMETERS:")
for attr, flag in sorted(FLAGS.__flags.items()):
  print("{} = {}".format(attr.upper(), flag.value))

if FLAGS.training_filepath==None:
  print("Input File path is empty. use --training_filepath argument.")
  exit()

max_document_length=15
inpH = InputHelper()
train_set, dev_set, vocab_processor, sum_no_of_batches = inpH.getDataSets(
  FLAGS.training_filepath, FLAGS.y_position, FLAGS.x1_position, FLAGS.x2_position, max_document_length, 10, FLAGS.batch_size)

trainableEmbeddings=False
if FLAGS.word2vec_model==None:
  trainableEmbeddings=True
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "You are using word embedding based semantic similarity but "
    "word2vec model path is empty. It is Recommended to use  --word2vec_model  argument. "
    "Otherwise now the code is automatically trying to learn embedding values (may not help in accuracy)"
    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
else:
  inpH.loadW2V(FLAGS.word2vec_model, FLAGS.word2vec_format)

# Training
# ==================================================
with tf.Graph().as_default():

  sess_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
  sess = tf.Session(config=sess_conf)

  with sess.as_default():
    siameseModel = SiameseLSTMw2v(
      sequence_length=max_document_length,
      vocab_size=len(vocab_processor.vocabulary_),
      embedding_size=FLAGS.embedding_dim,
      batch_size=FLAGS.batch_size,
      trainableEmbeddings=trainableEmbeddings,
      tied=FLAGS.tied,
      side1_nodes=FLAGS.side1_nodes,
      side2_nodes=FLAGS.side2_nodes,
    )
    
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
  
  grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
  train_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  # Keep track of gradient values and sparsity (optional)
  grad_summaries = []
  for g, v in grads_and_vars:
    if g is not None:
      grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
      sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
      grad_summaries.append(grad_hist_summary)
      grad_summaries.append(sparsity_summary)
  grad_summaries_merged = tf.summary.merge(grad_summaries)

  # Output directory for models and summaries
  out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time())))) \
            if FLAGS.output_dirpath is None else \
            os.path.abspath(FLAGS.output_dirpath)
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
  print("Writing to %s." % out_dir)

  # Summaries for loss pcc rho mse
  loss_summary = tf.summary.scalar("loss", siameseModel.loss)
  pcc_summary = tf.summary.scalar("pcc", siameseModel.pcc)
  rho_summary = tf.summary.scalar("rho", siameseModel.rho)
  mse_summary = tf.summary.scalar("mse", siameseModel.mse)

  # Train Summaries
  train_summary_op = tf.summary.merge([loss_summary, pcc_summary, rho_summary, mse_summary, grad_summaries_merged])
  train_summary_dir = os.path.join(out_dir, "summaries", "train")
  train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

  # Dev summaries
  dev_summary_op = tf.summary.merge([loss_summary, pcc_summary, rho_summary, mse_summary])
  dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
  dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

  # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
  checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

  # Write vocabulary
  vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  graph_def = tf.get_default_graph().as_graph_def()
  graphpb_txt = str(graph_def)
  with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
    f.write(graphpb_txt)

  if FLAGS.word2vec_model :
    # initial matrix with random uniform
    initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
    #initW = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))

    # load any vectors from the word2vec
    for w in vocab_processor.vocabulary_._mapping:
      arr=[]
      s = re.sub('[^0-9a-zA-Z]+', '', w)
      if w in inpH.pre_emb:
        arr=inpH.pre_emb[w]
      elif w.lower() in inpH.pre_emb:
        arr=inpH.pre_emb[w.lower()]
      elif s in inpH.pre_emb:
        arr=inpH.pre_emb[s]
      elif s.isdigit():
        arr=inpH.pre_emb["zero"]
      if len(arr)>0:
        idx = vocab_processor.vocabulary_.get(w)
        initW[idx]=np.asarray(arr).astype(np.float32)
    inpH.deletePreEmb()
    gc.collect()
    sess.run(siameseModel.W.assign(initW))

  def train_step(x1_batch, x2_batch, y_batch, i):
    random_value = random()
    feed_dict = {
      siameseModel.input_x1: x1_batch if random_value > 0.5 else x2_batch,
      siameseModel.input_x2: x2_batch if random_value > 0.5 else x1_batch,
      siameseModel.input_y_norm: map(lambda x: x / FLAGS.y_scale, y_batch),
      siameseModel.side1_dropout: FLAGS.side1_dropout,
      siameseModel.side2_dropout: FLAGS.side2_dropout,
    }

    _, step, loss, pcc, rho, mse, dist, summaries = sess.run([train_op_set, global_step, siameseModel.loss, siameseModel.pcc, siameseModel.rho, siameseModel.mse, siameseModel.distance, train_summary_op], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    if i % 100 == 0:
      print("TRAIN {}: step {}, loss {}, pcc: {}, rho: {}, mse: {}".format(time_str, step, loss, pcc, rho, mse * FLAGS.y_scale))
    train_summary_writer.add_summary(summaries, step)

  def dev_step(x1_batch, x2_batch, y_batch, i):
    random_value = random()
    feed_dict = {
      siameseModel.input_x1: x1_batch if random_value > 0.5 else x2_batch,
      siameseModel.input_x2: x2_batch if random_value > 0.5 else x1_batch,
      siameseModel.input_y_norm: map(lambda x: x / FLAGS.y_scale, y_batch),
      siameseModel.side1_dropout: 1.0,
      siameseModel.side2_dropout: 1.0,
    }
    
    step, loss, pcc, rho, mse, summaries = sess.run([global_step, siameseModel.loss, siameseModel.pcc, siameseModel.rho, siameseModel.mse, dev_summary_op], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    if i % 100 == 0:
      print("DEV {}: step {}, loss {}, pcc {}, rho {}, mse: {}".format(time_str, step, loss, pcc, rho, mse * FLAGS.y_scale))
    dev_summary_writer.add_summary(summaries, step)
    return mse * FLAGS.y_scale

  # Generate batches
  batches = inpH.batch_iter(list(zip(train_set[0], train_set[1], train_set[2])), FLAGS.batch_size, FLAGS.num_epochs)
  max_validation_mse=0.0

  n_iterations = sum_no_of_batches * FLAGS.num_epochs
  n_iterations = n_iterations if n_iterations < FLAGS.max_iterations else FLAGS.max_iterations

  print('Total number of iterations %s.' % n_iterations)
  for nn in xrange(n_iterations):
    batch = batches.next()
    if len(batch)<1:
      continue
    x1_batch, x2_batch, y_batch = zip(*batch)
    if len(y_batch)<1:
      continue
    train_step(x1_batch, x2_batch, y_batch, nn)
    step = tf.train.global_step(sess, global_step)

    current_evaluation_total_mse = 0.0

    if step % FLAGS.evaluate_every == 0:
      print("\nEvaluation:")
      dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), FLAGS.batch_size, 1)
      i = 0
      for db in dev_batches:
        if len(db)<1:
          continue
        x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
        if len(y_dev_b)<1:
          continue
        current_evaluation_total_mse = current_evaluation_total_mse + dev_step(x1_dev_b, x2_dev_b, y_dev_b, i)
        i = i + 1

    if step % FLAGS.checkpoint_every == 0 and current_evaluation_total_mse >= max_validation_mse:
      max_validation_mse = current_evaluation_total_mse
      saver.save(sess, checkpoint_prefix, global_step=step)
      tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
      print("Saved model {} with sum_mse={} checkpoint to {}\n".format(nn, max_validation_mse, checkpoint_prefix))
