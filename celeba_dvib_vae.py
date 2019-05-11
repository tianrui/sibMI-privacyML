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
import components as dvib

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

# Environment variables to set GPU/CPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.flags
logging = tf.logging
FLAGS = dvib.FLAGS

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
    #return FLAGS.beta * get_vae_cost(mean, stddev) + tf.reduce_sum(tf.pow(target_tensor - output_tensor, 2))
    return FLAGS.beta * get_vae_cost(mean, stddev) + tf.reduce_sum(-tf.cast(target_tensor, tf.float32) * tf.log(output_tensor + epsilon))


if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "celeba")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    model_directory = os.path.join(FLAGS.working_directory, "celeba_checkpoints")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)


    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 218, 178, 3])
    label_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])

    attrs = np.loadtxt(FLAGS.working_directory + FLAGS.attrs_directory + 'list_attr_celeba.txt', skiprows=2, usecols=range(1,FLAGS.output_size + FLAGS.private_size+1))
    def get_feed(batch_no, test_phase):
        xs = dvib.read_imgs(batch_no, test_phase)
        ys = dvib.read_attrs(attrs, batch_no, test_phase)
        return {input_tensor: xs, label_tensor: ys[:, :FLAGS.output_size], private_tensor: ys[:, FLAGS.output_size:]}

    with pt.defaults_scope(activation_fn=tf.nn.relu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvib.encoder(input_tensor)
            with tf.variable_scope("predictor") as scope:
                output_tensor, mean, stddev, epsilon = dvib.predictor(private_tensor, z)

        #with pt.defaults_scope(phase=pt.Phase.test):
        #    with tf.variable_scope("predictor", reuse=True) as scope:
        #        sampled_tensor, _, _ = predictor(private_tensor)

    vae_loss = get_vae_cost(mean, stddev)
    #rec_loss = get_reconstruction_cost(output_tensor, input_tensor)

    #loss = vae_loss + rec_loss
    loss = get_dvib_cost(mean, stddev, output_tensor, label_tensor)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.sign(output_tensor-0.5), tf.cast(tf.sign(label_tensor), tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[loss])
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement=True

    with tf.Session(config=config) as sess:
        sess.run(init)

        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                feeds = get_feed(i, False)
                pdb.set_trace()
                mean_val, dev_val, output_vals, vae_loss_val, eps_val, z_val = sess.run([mean, stddev, output_tensor, vae_loss, epsilon, z], feeds)
                _, loss_value, acc = sess.run([train, loss, accuracy], feeds)
                training_loss += loss_value

            training_loss = training_loss / \
                (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)

            print("Loss %f" % training_loss)
            print('Training Accuracy at epoch %s: %s' % (epoch, acc))

            #imgs = sess.run(sampled_tensor)
            #for k in range(FLAGS.batch_size):
            #    imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
            #    if not os.path.exists(imgs_folder):
            #        os.makedirs(imgs_folder)
            #
            #    imsave(os.path.join(imgs_folder, '%d.png') % k,
            #           imgs[k].reshape(28, 28))

            #pdb.set_trace()
            # Save model
            if epoch % 10 == 0:
                acc = 0
                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    acc += sess.run(accuracy, get_feed(i, True))

                acc /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                print('Test Accuracy at epoch %s: %s' % (epoch, acc))
                savepath = saver.save(sess, model_directory + '/dvib_celeba_model', global_step=epoch)
                print('Model saved at epoch %s: %s, path is %s' % (epoch, acc, savepath))
