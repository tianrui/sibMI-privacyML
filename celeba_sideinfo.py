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
import celeba_dvib_vae

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1600, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 200, "max epoch")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_string("working_directory", "/home/ubuntu", "")
flags.DEFINE_string("imgs_directory", "/data/celeba/img_align_celeba/", "")
flags.DEFINE_string("attrs_directory", "/data/celeba/", "")
flags.DEFINE_integer("hidden_size", 1024, "size of the hidden VAE unit")
flags.DEFINE_integer("z_size", 256, "size of the latent variable")
flags.DEFINE_integer("output_size", 20, "size of the output label")
flags.DEFINE_integer("private_size", 10, "size of the output label")
flags.DEFINE_integer("side_size", 10, "size of the side information vector")
flags.DEFINE_float("beta", 1e-3, "Beta factor for KL divergence in DVIB")
flags.DEFINE_float("keep_prob", 0.8, "Probability of keeping a hidden unit with dropout")
flags.DEFINE_integer("sample_size", 12, "Number of samples of latent variable to average over")
flags.DEFINE_float("max_stddev", 80, "Maximum allowed standard deviation when sampling, to prevent overflow/underflow")
flags.DEFINE_integer("dataset_size", 160000, "size of the dataset")
flags.DEFINE_integer("test_dataset_size", 20000, "size of the dataset")

FLAGS = flags.FLAGS


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
    #return get_vae_cost(mean, stddev) + tf.reduce_sum(-tf.cast(target_tensor, tf.float32) * tf.log(output_tensor + epsilon))
    return FLAGS.beta * get_vae_cost(mean, stddev) + tf.reduce_sum(tf.pow(target_tensor - output_tensor, 2))

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
        xs = read_imgs(batch_no, test_phase)
        ys = read_attrs(attrs, batch_no, test_phase)
        return {input_tensor: xs, label_tensor: ys[:, :FLAGS.output_size], private_tensor: ys[:, FLAGS.output_size:]}

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = encoder(input_tensor)
            with tf.variable_scope("predictor") as scope:
                output_tensor, mean, stddev, epsilon = predictor(private_tensor, z)

        #with pt.defaults_scope(phase=pt.Phase.test):
        #    with tf.variable_scope("predictor", reuse=True) as scope:
        #        sampled_tensor, _, _ = predictor(private_tensor)

    vae_loss = get_vae_cost(mean, stddev)
    #rec_loss = get_reconstruction_cost(output_tensor, input_tensor)

    #loss = vae_loss + rec_loss
    loss = get_dvib_cost(mean, stddev, output_tensor, label_tensor)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.sign(output_tensor), tf.cast(tf.sign(label_tensor), tf.float32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[loss])
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()

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
                #pdb.set_trace()
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
