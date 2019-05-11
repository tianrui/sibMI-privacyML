from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
import pdb
import matplotlib
matplotlib.use('Agg')
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


def encoder(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 218*178*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            flatten().
            fully_connected(FLAGS.hidden_size * 1, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size * 1, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.z_size * 2, activation_fn=None)).tensor

def side_encoder(input_tensor, side_tensor=None):
    '''Create side information encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 218*178*3]
        side_tensor: a batch of side information vectors [batch_size, side_size]

    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            flatten().
            concat(1, [tf.cast(side_tensor, tf.float32)]).
            fully_connected(FLAGS.hidden_size * 2, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size * 1, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.z_size * 2, activation_fn=None)).tensor

def privacy_encoder(input_tensor, private_tensor):
    '''Create side information encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, input_size]
        side_tensor: a batch of side information vectors [batch_size, side_size]

    Returns:
        A tensor that expresses the encoder network
    '''
    tmp = (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, np.sqrt(FLAGS.input_size), np.sqrt(FLAGS.input_size), 1]).
            conv2d(5, 32, stride=2, activation_fn=tf.nn.sigmoid).
            max_pool(5, stride=2).
            conv2d(5, 64, stride=2, activation_fn=tf.nn.sigmoid).
            max_pool(5, stride=2).
            dropout(0.9).
            flatten().
            concat(1, [tf.cast(private_tensor, tf.float32)]).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            fully_connected(int(FLAGS.hidden_size/2), activation_fn=tf.nn.relu).
            fully_connected(FLAGS.z_size * 2, activation_fn=None)).tensor

    return tmp

def synth_encoder(input_tensor, private_tensor, hidden_size):
    '''Create encoder for synthetic data mapping
       input to a latent representation with a Gaussian distribution
    '''
    #pdb.set_trace()
    tmp = (pt.wrap(input_tensor).
            concat(1, [tf.cast(private_tensor, tf.float32)]).
            fully_connected(hidden_size, activation_fn=tf.nn.relu).
            fully_connected(int(hidden_size/2), activation_fn=tf.nn.relu).
            fully_connected(FLAGS.z_size*2, activation_fn=None)).tensor
    return tmp

def synth_affine_encoder(input_tensor, private_tensor, beta0, beta1):
    ''' Create an encoder with affine transformation
        of the input data x, for two class data
    '''
    z = input_tensor + beta0 * private_tensor + beta1 * (1.0 - private_tensor)
    return z

def synth_affine_noisy_encoder(input_tensor, private_tensor, beta0, beta1, gamma0, gamma1):
    ''' Create an encoder with affine transformation and class dependent noise
        of the input data x, for two class data
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.z_size])
    z = input_tensor + beta0 * private_tensor + beta1 * (1.0 - private_tensor) + \
            gamma0 * epsilon * private_tensor + gamma1 * epsilon * (1.0 - private_tensor)
    return z

def synth_affine_indepnoisy_encoder(input_tensor, private_tensor, beta0, beta1, gamma):
    ''' Create an encoder with affine transformation and class indep noise
        of the input data x, for two class data
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.z_size])
    z = input_tensor + beta0 * private_tensor + beta1 * (1.0 - private_tensor) + \
            gamma * epsilon
    return z

def mnist_encoder(input_tensor):
    '''Create encoder network for MNIST data with convolutional nn.

    Argse
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor

def pendigits_encoder(input_tensor):
    '''Create encoder network for pendigits dataset to replicate arxiv 1802.09386
        encoder has 700 units with 8 layers, dropout of 0.1, ReLU activations.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    FLAGS.hidden_size=700
    FLAGS.keep_prob=0.9
    return (pt.wrap(input_tensor).
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
            fully_connected(FLAGS.z_size, activation_fn=tf.nn.relu)).tensor

def ferg_encoder(input_tensor, private_tensor):
    '''Create encoder network for FERG dataset to replicate arxiv 1802.09386
        encoder has 1200 units with 8 layers, dropout of 0.1, ReLU activations.

    Args:
        input_tensor: a batch of flattened images [batch_size, 50*50]

    Returns:
        A tensor that expresses the encoder network
    '''
    #FLAGS.z_size=1200
    #FLAGS.hidden_size=1200
    FLAGS.keep_prob=0.9
    return (pt.wrap(input_tensor).
            concat(1, [tf.cast(private_tensor, tf.float32)]).
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
            fully_connected(FLAGS.z_size*2, activation_fn=None)).tensor
 
def decoder(input_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.z_size])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:]))
        input_sample = mean + epsilon * stddev
    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.z_size]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()).tensor, mean, stddev

def predictor(private_tensor, input_tensor=None):
    """Create prediction network
    """
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
    input_sample = tf.reduce_mean(input_sample, axis=0)
    return (pt.wrap(input_sample).
            concat(1, [tf.cast(private_tensor, tf.float32)]).
            fully_connected(FLAGS.hidden_size * 2, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.output_size, activation_fn=tf.nn.sigmoid)).tensor, mean, stddev, epsilon

def private_predictor(input_tensor=None):
    """Create a predictor for private labels and public labels
    """
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
    input_sample = tf.reduce_mean(input_sample, axis=0)
    return (pt.wrap(input_sample).
            fully_connected(FLAGS.hidden_size * 2, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.output_size + FLAGS.private_size, activation_fn=tf.nn.sigmoid)).tensor, mean, stddev, epsilon

def privacy_reconstructor(input_tensor=None):
    """Create a predictor for private labels and public images
    xhat: tensor of reconstructed image (normalized between 0 and 1)
    chat: tensor of private predictions (probabilities)
    """
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    input_tensor = tf.cast(input_tensor, tf.float32)
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
        # Use deterministic mapping
        #stddev = None
        #input_sample = mean
    input_sample = tf.reduce_mean(input_sample, axis=0)
    xhat = (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.z_size]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 8, stride=2, activation_fn=tf.nn.relu).
            flatten().
            fully_connected(FLAGS.img_height * FLAGS.img_width * 3, activation_fn=tf.nn.sigmoid).
            reshape([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3])).tensor

    chat = (pt.wrap(input_sample).
            fully_connected(FLAGS.hidden_size * 2, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.private_size, activation_fn=None)).tensor
    return xhat, chat, mean, epsilon

def synth_predictor(input_tensor, sampling=True):
    '''Create a predictor for synthetic data to map to outputs
    '''
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    input_tensor = tf.cast(input_tensor, tf.float32)
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
    if sampling:
        input = tf.reduce_mean(input_sample, axis=0)
    else:
        input = input_tensor
    # Switch between input_sample and input_tensor to add/remove noise
    chat = (pt.wrap(input).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            fully_connected(int(FLAGS.hidden_size/2), activation_fn=tf.nn.relu).
            fully_connected(FLAGS.private_size, activation_fn=None)).tensor
    yhat = (pt.wrap(input).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            fully_connected(int(FLAGS.hidden_size/2), activation_fn=tf.nn.relu).
            fully_connected(FLAGS.output_size, activation_fn=None)).tensor
    return yhat, chat, mean, stddev

def mnist_predictor(input_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    input_tensor = tf.cast(input_tensor, tf.float32)
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
    input_sample = tf.reduce_mean(input_sample, axis=0)
    #original decoder takes in input tensor, returns yhat, mean, stddev
    #yhat = (pt.wrap(input_tensor[:, :FLAGS.z_size]).
    yhat = (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.z_size]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()).tensor
    
    #chat = (pt.wrap(input_tensor[:, :FLAGS.z_size]).
    chat = (pt.wrap(input_sample).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size/2, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.private_size, activation_fn=None)).tensor
    return yhat, chat, mean, stddev

def synth_twotask_predictor(input_tensor=None):
    '''Create decoder network to predict in two tasks for the synthetic database
       If input tensor is provided then decodes it, otherwise sample it
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        yhat A tensor that expresses the decoder network for regular task
        chat A tensor that expresses the decoder network for private task

    Replicating the arxiv 1802.09386 network:
        3 layers of 1200 units, dropout probability of 0.1 between layers
        ReLU activations except for last layer, output layer has softmax activations.
    '''
    FLAGS.keep_prob=0.9
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    input_tensor = tf.cast(input_tensor, tf.float32)
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
    input_sample = tf.reduce_mean(input_sample, axis=0)
    #original decoder takes in input tensor, returns yhat, mean, stddev
    yhat = (pt.wrap(input_tensor[:, :FLAGS.z_size]).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.output_size, activation_fn=tf.nn.softmax).
            flatten()).tensor
    
    chat = (pt.wrap(input_tensor[:, :FLAGS.z_size]).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.private_size, activation_fn=None)).tensor
    return yhat, chat, mean, stddev


def ferg_twotask_predictor(input_tensor=None):
    '''Create decoder network to predict in two tasks for the FERG database
       If input tensor is provided then decodes it, otherwise sample it
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        yhat A tensor that expresses the decoder network for regular task
        chat A tensor that expresses the decoder network for private task

    Replicating the arxiv 1802.09386 network:
        3 layers of 1200 units, dropout probability of 0.1 between layers
        ReLU activations except for last layer, output layer has softmax activations.
    '''
    # Get all the previous variables as encode parameters
    encode_params_len = len(tf.trainable_variables())
    #FLAGS.z_size=1200
    #FLAGS.hidden_size=1200
    FLAGS.keep_prob=0.9
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    input_tensor = tf.cast(input_tensor, tf.float32)
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, :FLAGS.z_size])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
    input_sample = tf.reduce_mean(input_sample, axis=0)
    #original decoder takes in input tensor, returns yhat, mean, stddev
    with tf.variable_scope("regular_decoder") as scope:
        #yhat = (pt.wrap(input_tensor[:, :FLAGS.z_size]).
        yhat = (pt.wrap(input_sample).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(int(FLAGS.hidden_size/2), activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.output_size, activation_fn=None).
            flatten()).tensor
    
    # All current parameters give us the regular branch parameters
    all_params = tf.trainable_variables()
    reg_params_len = len(all_params) - encode_params_len
    with tf.variable_scope("private_decoder") as scope:
        #chat = (pt.wrap(input_tensor[:, :FLAGS.z_size]).
        chat = (pt.wrap(input_sample).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(int(FLAGS.hidden_size/2), activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.private_size, activation_fn=None)).tensor
    # All current parameters give us the adversary branch parameters
    all_params = tf.trainable_variables()
    adv_params_len = len(all_params) - encode_params_len - reg_params_len
    return yhat, chat, mean, stddev, reg_params_len, adv_params_len

def ferg_generator(input_tensor=None):
    '''Create decoder network to predict in two tasks for the FERG database
       If input tensor is provided then decodes it, otherwise sample it
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        xhat A tensor expressing the reconstruction for x
        yhat A tensor that expresses the decoder network for regular task
        chat A tensor that expresses the decoder network for private task

    Replicating the arxiv 1802.09386 network:
        3 layers of 1200 units, dropout probability of 0.1 between layers
        ReLU activations except for last layer, output layer has softmax activations.
    '''
    # Get all the previous variables as encode parameters
    encode_params_len = len(tf.trainable_variables())
    FLAGS.input_size=2500
    FLAGS.keep_prob=0.9
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    input_tensor = tf.cast(input_tensor, tf.float32)
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.clip_by_value(tf.sqrt(tf.exp(input_tensor[:, :FLAGS.z_size])), -FLAGS.max_stddev, FLAGS.max_stddev)
        input_sample = mean + epsilon * stddev
    input_sample = tf.reduce_mean(input_sample, axis=0)
    #original decoder takes in input tensor, returns xhat, yhat, mean, stddev
    xhat = (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.z_size]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.relu).
            flatten().
            fully_connected(FLAGS.input_size, activation_fn=tf.nn.sigmoid).
            reshape([FLAGS.batch_size, FLAGS.input_size])).tensor
    
    yhat = (pt.wrap(input_sample).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.reg_size, activation_fn=tf.nn.softmax).
            flatten()).tensor
    # All current parameters give us the reconstruction parameters
    all_params = tf.trainable_variables()
    recon_params_len = len(all_params) - encode_params_len
    chat = (pt.wrap(input_sample).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(FLAGS.keep_prob).
            fully_connected(FLAGS.private_size, activation_fn=None)).tensor
    # Current params gives us the adversary params
    all_params = tf.trainable_variables()
    adv_params_len = len(all_params) - encode_params_len - recon_params_len
    return xhat, yhat, chat, mean, stddev, recon_params_len, adv_params_len

def mnist_discriminator(input_tensor):
    '''Create a discriminator network for synthetic data to classify
        whether image was cleaned or not
        input_tensor: input tensor of NxM
        output_vec: output tensor of binary probabilities N
    '''
    yhat = (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            fully_connected(1, activation_fn=None)).tensor
    return yhat

def ferg_discriminator(input_tensor):
    '''Create a discriminator network for FERG data to classify
        whether image was cleaned or not
        input_tensor: input tensor of NxM
        output_vec: output tensor of binary probabilities N
    '''
    yhat = (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 50, 50, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(0.9).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            dropout(0.9).
            fully_connected(1, activation_fn=None)).tensor
    return yhat

def synth_discriminator(input_tensor):
    '''Create a discriminator network for synthetic data to classify
        whether data was cleaned or not
        input_tensor: input tensor of NxM
        output_vec: output tensor of binary probabilities N
    '''
    yhat = (pt.wrap(input_tensor).
            flatten().
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            fully_connected(FLAGS.hidden_size/2, activation_fn=tf.nn.relu).
            fully_connected(1, activation_fn=None)).tensor
    return yhat

def read_imgs(batch_no, test_phase):
    imgs = np.zeros((FLAGS.batch_size, 218, 178, 3))
    if test_phase == True:
        trainset_offset = FLAGS.dataset_size
    else:
        trainset_offset = 0
    for i in np.arange(0, FLAGS.batch_size):
        imgs[i] = plt.imread(FLAGS.working_directory + FLAGS.imgs_directory + str(trainset_offset + batch_no * FLAGS.batch_size + i + 1).rjust(6, '0') + '.jpg', format='jpg')
    return imgs

def read_attrs(attrs, batch_no, test_phase):
    if test_phase == True:
        trainset_offset = FLAGS.dataset_size
    else:
        trainset_offset = 0
    attrs_batch = np.array(attrs[trainset_offset + batch_no*FLAGS.batch_size : trainset_offset + (batch_no+1)*FLAGS.batch_size])
    return attrs_batch


