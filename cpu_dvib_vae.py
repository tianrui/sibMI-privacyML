'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

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

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 600, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 200, "max epoch")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 1024, "size of the hidden VAE unit")
flags.DEFINE_integer("z_size", 256, "size of the latent variable")
flags.DEFINE_integer("output_size", 10, "size of the output label")
flags.DEFINE_float("beta", 1e-3, "Beta factor for KL divergence in DVIB")
flags.DEFINE_float("sample_size", 12, "Number of samples of latent variable to average over")

FLAGS = flags.FLAGS


def encoder(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            flatten().
            fully_connected(FLAGS.hidden_size * 2, activation_fn=tf.nn.relu).
            fully_connected(FLAGS.hidden_size * 2, activation_fn=tf.nn.relu).
            fully_connected(FLAGS.z_size * 2, activation_fn=None)).tensor


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

def predictor(input_tensor=None):
    '''Create prediction network
    '''
    epsilon = tf.random_normal([FLAGS.sample_size, FLAGS.batch_size, FLAGS.z_size])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.z_size]
        stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.z_size:]))
        input_sample = mean + epsilon * stddev
    input_sample = tf.reduce_mean(input_sample, axis=0)
    return (pt.wrap(input_sample).
            fully_connected(FLAGS.output_size, activation_fn=None).
            softmax_activation()).tensor, mean, stddev, epsilon
            
 

def get_vae_cost(mean, stddev, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean: 
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder 
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''
    return tf.reduce_sum(-tf.cast(target_tensor, tf.float32) * tf.log(output_tensor + epsilon) -
                         (1.0 - tf.cast(target_tensor, tf.float32)) * tf.log(1.0 - output_tensor + epsilon))

def get_dvib_cost(mean, stddev, output_tensor, target_tensor, epsilon=1e-8):
    '''DVIB loss
    Args:
        mean
        stddev
        output_tensor: tensor produces by decoder 
        target_tensor: the target tensor that we want to reconstruct
        epsilon
    '''
    #return get_vae_cost(mean, stddev) + get_reconstruction_cost(output_tensor, target_tensor) 
    return FLAGS.beta * get_vae_cost(mean, stddev) + tf.reduce_sum(-tf.cast(target_tensor, tf.float32) * tf.log(output_tensor + epsilon))

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    model_directory = os.path.join(FLAGS.working_directory, "checkpoints")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
 
    mnist = input_data.read_data_sets(data_directory, one_hot=True)

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])
    label_tensor = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.output_size])

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = encoder(input_tensor)
            with tf.variable_scope("decoder") as scope:
                output_tensor, mean, stddev, epsilon = predictor(z)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("decoder", reuse=True) as scope:
                sampled_tensor, _, _, _ = predictor()

    vae_loss = get_vae_cost(mean, stddev)
    #rec_loss = get_reconstruction_cost(output_tensor, input_tensor)

    #loss = vae_loss + rec_loss
    loss = get_dvib_cost(mean, stddev, output_tensor, label_tensor)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(output_tensor, 1), tf.argmax(label_tensor, 1))
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[loss])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        # Init lists to store perf data
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, y = mnist.train.next_batch(FLAGS.batch_size)
                mean_val, dev_val, output_vals, vae_loss_val, eps_val, z_val = sess.run([mean, stddev, output_tensor, vae_loss, epsilon, z], {input_tensor: x, label_tensor: y})
                _, loss_value, vae_loss_val, acc = sess.run([train, loss, vae_loss, accuracy], {input_tensor: x, label_tensor: y})
                training_loss += loss_value

            training_loss = training_loss / \
                (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)

            print("Loss %f" % training_loss)
            print('Training Accuracy at epoch %s: %s' % (epoch, acc))

            train_loss_list.append(training_loss)
            train_acc_list.append(acc)

            imgs = sess.run(sampled_tensor)
            #for k in range(FLAGS.batch_size):
            #    imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
            #    if not os.path.exists(imgs_folder):
            #        os.makedirs(imgs_folder)
            #    
            #    imsave(os.path.join(imgs_folder, '%d.png') % k,
            #           imgs[k].reshape(28, 28))

            # Save model
            if epoch % 10 == 0:
                acc = 0
                for i in range(int(mnist.test._num_examples / FLAGS.batch_size)):
                    xtest, ytest = mnist.test.next_batch(FLAGS.batch_size)
                    acc += sess.run(accuracy, {input_tensor: xtest,
                                label_tensor: ytest})
                
                acc /= int(mnist.test._num_examples / FLAGS.batch_size)
                print('Test Accuracy at epoch %s: %s' % (epoch, acc))
                test_acc_list.append(acc)
                saver.save(sess, model_directory + '/dvib_mnist_model', global_step=epoch)

        np.savez(model_directory+'/train_perf', train_loss=np.array(train_loss_list),
                train_acc=np.array(train_acc_list),
                test_acc=np.array(test_acc_list))
