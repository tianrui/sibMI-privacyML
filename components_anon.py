from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
#from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

# Environment variables to set GPU/CPU device
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1600, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 200, "max epoch")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("beta", 1e-3, "Beta factor for KL divergence in DVIB")
flags.DEFINE_float("keep_prob", 0.8, "Probability of keeping a hidden unit with dropout")
flags.DEFINE_integer("sample_size", 12, "Number of samples of latent variable to average over")
flags.DEFINE_float("max_stddev", 80, "Maximum allowed standard deviation when sampling, to prevent overflow/underflow")
flags.DEFINE_integer("dataset_size", 160000, "size of the dataset")
flags.DEFINE_integer("test_dataset_size", 20000, "size of the dataset")

flags.DEFINE_string("working_directory", "/home/rxiao", "")
flags.DEFINE_string("summary_dir", "/home/rxiao", "")
flags.DEFINE_string("imgs_directory", "/data/celeba/img_align/img_align_celeba/", "")
flags.DEFINE_string("attrs_directory", "/data/celeba/", "")

flags.DEFINE_integer("img_width", 178, "width of image in celebA dataset")
flags.DEFINE_integer("img_height", 218, "height of image in celebA dataset")

flags.DEFINE_integer("input_size", 178*218*3, "size of the input label")
flags.DEFINE_integer("hidden_size", 1024, "size of the hidden VAE unit")
flags.DEFINE_integer("z_size", 128, "size of the latent variable")
flags.DEFINE_integer("output_size", 20, "size of the output label")
flags.DEFINE_integer("private_size", 10, "size of the output label")
flags.DEFINE_integer("side_size", 10, "size of the side information")

FLAGS = flags.FLAGS

'''Function components to implement anonymized representations with adversarial networks
arxiv: 1802.09386
'''
def anon_encoder_digits(input_tensor):
    '''Create encoder network for pen_digits/mnist experiment
    8 layers of 700 units each, relu activations, dropout with keep_prob=0.9
    '''
    FLAGS.hidden_size=700
    FLAGS.keep_prob=0.9
    return (pt.wrap(input_tensor).
            flatten().
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu)).tensor

def anon_reg_decoder_digits(input_tensor):
    '''Create encoder network for pen_digits/mnist experiment
    3 layers of 700 units each, relu activations, dropout with keep_prob=0.9
    Regular task is to recognize the digit, out of 10
    '''
    FLAGS.hidden_size=700
    FLAGS.keep_prob=0.9
    FLAGS.output_size=10
    return (pt.wrap(input_tensor).
            flatten().
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.softmax)).tensor

def anon_priv_decoder_digits(input_tensor):
    '''Create encoder network for pen_digits/mnist experiment
    3 layers of 700 units each, relu activations, dropout with keep_prob=0.9
    Private task is to identify the person, out of 30
    '''
    FLAGS.hidden_size=700
    FLAGS.keep_prob=0.9
    FLAGS.private_size=30
    return (pt.wrap(input_tensor).
            flatten().
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.softmax)).tensor

