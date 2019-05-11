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
#Use model file to restore model
import celeba_dvib_vae as dvib

FLAGS = dvib.FLAGS
FLAGS.max_epoch = 100
FLAGS.keep_prob = 1.0

def test_predictor(input_tensor):
    '''Create prediction network for both public and private labels
    '''
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

    return (pt.wrap(input_tensor).
            fully_connected(FLAGS.hidden_size * 2, activation_fn=tf.nn.relu).
            fully_connected(FLAGS.hidden_size, activation_fn=tf.nn.relu).
            fully_connected(FLAGS.output_size + FLAGS.private_size, activation_fn=None)).tensor

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "celebA")
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


    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvib.encoder(input_tensor)
            with tf.variable_scope("predictor") as scope:
                output_tensor, mean, stddev, epsilon = dvib.predictor(private_tensor, z)
            with tf.variable_scope("test_predictor") as scope:
                full_predictions = test_predictor(z)

        #with pt.defaults_scope(phase=pt.Phase.test):
        #    with tf.variable_scope("predictor", reuse=True) as scope:
        #        sampled_tensor, _, _ = predictor(private_tensor)
    params2 = tf.trainable_variables()
    vae_loss = dvib.get_vae_cost(mean, stddev)
    #rec_loss = get_reconstruction_cost(output_tensor, input_tensor)

    #loss = vae_loss + rec_loss
    #loss = dvib.get_dvib_cost(mean, stddev, output_tensor, label_tensor)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.concat([label_tensor, private_tensor], 1), logits=full_predictions))
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_correct_prediction'):
            pub_correct_prediction = tf.equal(tf.sign(full_predictions[:, :FLAGS.output_size]), tf.cast(tf.sign(label_tensor), tf.float32))
        with tf.name_scope('pub_accuracy'):
            pub_accuracy = tf.reduce_mean(tf.cast(pub_correct_prediction, tf.float32))
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_correct_prediction'):
            sec_correct_prediction = tf.equal(tf.sign(full_predictions[:, FLAGS.output_size:]), tf.cast(tf.sign(private_tensor), tf.float32))
        with tf.name_scope('sec_accuracy'):
            sec_accuracy = tf.reduce_mean(tf.cast(sec_correct_prediction, tf.float32))


    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    test_train = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="test_predictor"))
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder")
    restore_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder"))
    saver = tf.train.Saver(restore_vars)
    init = tf.initialize_all_variables()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement=True

    sess = tf.Session(config=config)
    checkpt = tf.train.latest_checkpoint(model_directory)
    saver.restore(sess, checkpt)
    print("Restored encoder module from checkpoint %s" % (checkpt))
    sess.run(init)
    #pdb.set_trace()

    for epoch in range(FLAGS.max_epoch):
        training_loss = 0.0

        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
        pbar.start()
        pub_acc = 0
        sec_acc = 0
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, False)
            #mean_val, dev_val, output_vals, vae_loss_val, eps_val, z_val = sess.run([mean, stddev, output_tensor, vae_loss, epsilon, z], feeds)
            pub_corr, sec_corr = sess.run([pub_correct_prediction, sec_correct_prediction], feeds)
            _, loss_value, pub_tmp, sec_tmp = sess.run([test_train, loss, pub_accuracy, sec_accuracy], feeds)
            training_loss += loss_value
            pub_acc += pub_tmp
            sec_acc += sec_tmp

        training_loss = training_loss / \
            (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)
        pub_acc /= FLAGS.updates_per_epoch
        sec_acc /= FLAGS.updates_per_epoch

        print("Loss %f" % training_loss)
        print('Training public Accuracy at epoch %s: %s' % (epoch, pub_acc))
        print('Training private Accuracy at epoch %s: %s' % (epoch, sec_acc))
        if epoch % 10 == 9:
                pubacc = 0
                secacc = 0
                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    pubacc += sess.run(pub_accuracy, get_feed(i, True))
                    secacc += sess.run(sec_accuracy, get_feed(i, True))

                pubacc /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                secacc /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                print('Test public Accuracy at epoch %s: %s' % (epoch, pubacc))
                print('Test private Accuracy at epoch %s: %s' % (epoch, secacc))
                savepath = saver.save(sess, model_directory + '/dvib_celeba_modeltest', global_step=epoch)
                print('Model saved at epoch %s, path is %s' % (epoch, savepath))


    sess.close()


