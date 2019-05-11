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
from tensorflow.python.client import device_lib

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar
#Use model file to restore model
import components as dvib

# Environment variables to set GPU/CPU device
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#dvib.FLAGS.DEFINE_float("encode_coef", 1e-3, "Lambda coef for l2 term in encoding cost")
FLAGS = dvib.FLAGS
FLAGS.working_directory = "/home/ubuntu"
FLAGS.batch_size = 160
FLAGS.updates_per_epoch = 100
FLAGS.max_epoch = 1000
FLAGS.keep_prob = 1.0
FLAGS.z_size = 521
FLAGS.output_size = 30
FLAGS.private_size = 10

PX_MAX = 255
eps = 1e-16

def list_devices():
    #pdb.set_trace()
    local_dev_protos = device_lib.list_local_devices()
    for x in local_dev_protos:
        print(x.name)
    return

def privacy_reconstruction_cost(output_tensor, output_tensor_priv, private_tensor, target_tensor, decode_coef=1):
    '''Cost of model decoder to reconstruct the target tensor and the private tensor
    '''
    #l2cost_private = tf.reduce_mean((output_tensor_priv - private_tensor)**2)
    real_logits = -tf.log(tf.maximum(1.0 / output_tensor_priv - 1.0, eps))
    l2cost_private = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=private_tensor, logits=real_logits))
    l2cost_x = tf.reduce_mean((output_tensor - target_tensor)**2)
    return l2cost_private + decode_coef * l2cost_x

def privacy_encoding_cost(output_tensor, output_tensor_priv, private_tensor, target_tensor, prior_tensor, encode_coef=1e-3):
    '''Cost of model encoder to encode input and private tensor into a latent representation
    p_c: p(c=1) in each dim
    p_cz: posterior distribution predicted based on z, p(c=1|z) for each dim
    '''
    #p_ci = tf.cast(private_tensor, tf.float32) * output_tensor_priv
    #KL_pcz_pc = tf.reduce_mean(p_ci * tf.log(p_ci))
    p_cz = output_tensor_priv
    p_c = prior_tensor
    #KL_pcz_pc = tf.reduce_mean(p_cz * tf.log(p_cz)) - tf.reduce_mean(p_cz * tf.log(p_c + eps))
    KL_pcz_pc = tf.reduce_mean(p_cz * tf.log(tf.maximum(p_cz / p_c, eps)) + (1. - p_cz) * tf.log(tf.maximum((1. - p_cz) / (1. - p_c), eps)))
    l2cost_x = tf.reduce_mean((output_tensor - target_tensor)**2)
    l2cost_private = tf.reduce_mean((output_tensor_priv - private_tensor)**2)
    loss = KL_pcz_pc + encode_coef * l2cost_x
    #loss = -l2cost_private + encode_coef * l2cost_x
    return loss, KL_pcz_pc

def calc_pc():
    '''Calculate the prior probabilities of each dimension based on counts in a binary dataset
    data: N x M matrix of 1, -1
    returns pc: M x 1 vector of p(c=1)
    '''
    data_directory = os.path.join(FLAGS.working_directory, "celebA")
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    attrs = np.loadtxt(FLAGS.working_directory + FLAGS.attrs_directory + 'list_attr_celeba.txt', skiprows=2, usecols=range(1,FLAGS.output_size + FLAGS.private_size+1))
    def get_feed(batch_no, test_phase):
        #xs = dvib.read_imgs(batch_no, test_phase)
        ys = dvib.read_attrs(attrs, batch_no, test_phase)
        #xs = xs/PX_MAX
        #ys = ys * PX_MAX / 2
        return {private_tensor: ys[:, FLAGS.output_size:]}
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("prior") as scope:
                # private data is in {-1, 1}
                p_c = tf.reduce_mean((private_tensor+1.)/2., axis=0)

    #pdb.set_trace()
    init = tf.global_variables_initializer()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False
    sess = tf.Session()
    sess.run(init)
    # calculating prior p_c
    widgets = ["Calculating priors |", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
    pbar.start()


    p_cv = 0
    for i in range(FLAGS.updates_per_epoch):
        pbar.update(i)
        feeds = get_feed(i, False)
        p_cv += sess.run(p_c, feeds)
    p_cv /= FLAGS.updates_per_epoch
    print("prior p_c")
    print(p_cv)
    sess.close()
    return p_cv

def training_privacy(prior):
    data_directory = os.path.join(FLAGS.working_directory, "celebA")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    model_directory = os.path.join(data_directory, "privacy_checkpoints")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 218, 178, 3])
    label_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])

    attrs = np.loadtxt(FLAGS.working_directory + FLAGS.attrs_directory + 'list_attr_celeba.txt', skiprows=2, usecols=range(1,FLAGS.output_size + FLAGS.private_size+1))
    def get_feed(batch_no, test_phase):
        xs = dvib.read_imgs(batch_no, test_phase)
        ys = dvib.read_attrs(attrs, batch_no, test_phase)
        #xs = xs/PX_MAX
        #ys = ys * PX_MAX / 2
        return {input_tensor: xs, label_tensor: ys[:, :FLAGS.output_size], private_tensor: ys[:, FLAGS.output_size:]}


    with pt.defaults_scope(activation_fn=tf.nn.relu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvib.privacy_encoder(input_tensor, private_tensor)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder") as scope:
                xhat, chat, mean, epsilon = dvib.privacy_reconstructor(z)
                xhat = xhat * PX_MAX
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len

        #with pt.defaults_scope(phase=pt.Phase.test):
        #    with tf.variable_scope("predictor", reuse=True) as scope:
        #        sampled_tensor, _, _ = predictor(private_tensor)

    loss1, KLloss = privacy_encoding_cost(xhat, chat, private_tensor, input_tensor, prior_tensor)
    loss2 = privacy_reconstruction_cost(xhat, chat, private_tensor, input_tensor)
    full_loss = loss1 + loss2
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean(tf.reduce_sum((xhat - input_tensor)**2, [1, 2, 3]))
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            sec_dist = tf.reduce_mean((chat - private_tensor)**2)
            correct_pred = tf.less(tf.abs(chat - private_tensor), 0.5)
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    e_train = pt.apply_optimizer(optimizer, losses=[loss1], regularize=True, include_marked=True, var_list=encode_params)
    d_train = pt.apply_optimizer(optimizer, losses=[loss2], regularize=True, include_marked=True, var_list=all_params[e_param_len:])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False

    sess = tf.Session()
    sess.run(init)

    pub_loss_train = np.zeros(FLAGS.updates_per_epoch)
    sec_loss_train = np.zeros(FLAGS.updates_per_epoch)
    sec_acc_train = np.zeros(FLAGS.updates_per_epoch)
    e_loss_train = np.zeros(FLAGS.updates_per_epoch)
    d_loss_train = np.zeros(FLAGS.updates_per_epoch)
    KL_loss_train = np.zeros(FLAGS.updates_per_epoch)
    pub_loss_val = np.zeros(FLAGS.updates_per_epoch)
    sec_loss_val = np.zeros(FLAGS.updates_per_epoch)
    sec_acc_val = np.zeros(FLAGS.updates_per_epoch)
    e_loss_val = np.zeros(FLAGS.updates_per_epoch)
    d_loss_val = np.zeros(FLAGS.updates_per_epoch)
    KL_loss_val = np.zeros(FLAGS.updates_per_epoch)

    for epoch in range(FLAGS.max_epoch):
        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
        pbar.start()

        pub_loss = 0
        sec_loss = 0
        e_training_loss = 0
        d_training_loss = 0
        KLv = 0
        secaccv = 0

        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, False)
            #mean_val, dev_val, output_vals, vae_loss_val, eps_val, z_val = sess.run([mean, stddev, output_tensor, vae_loss, epsilon, z], feeds)
            #zv, xhatv, chatv, meanv, stddevv, epsilonv = sess.run([z, xhat, chat, mean, stddev, epsilon], feeds)
            pub_tmp, sec_tmp, secacc_tmp, KL_tmp = sess.run([pub_dist, sec_dist, sec_acc, KLloss], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            _, d_loss_value = sess.run([d_train, loss2], feeds)
            if np.isnan(e_loss_value) or np.isnan(d_loss_value):
                pdb.set_trace()
                break
            e_training_loss += e_loss_value
            d_training_loss += d_loss_value
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            KLv += KL_tmp
            secaccv += secacc_tmp

        e_training_loss = e_training_loss / \
            (FLAGS.updates_per_epoch)
        d_training_loss = d_training_loss / \
            (FLAGS.updates_per_epoch)
        pub_loss /= FLAGS.updates_per_epoch
        sec_loss /= FLAGS.updates_per_epoch
        secaccv /= FLAGS.updates_per_epoch
        KLv /= FLAGS.updates_per_epoch

        print("Loss for E %f, and for D %f" % (e_training_loss, d_training_loss))
        print('Training public loss at epoch %s: %s' % (epoch, pub_loss))
        print('Training private loss at epoch %s: %s' % (epoch, sec_loss))
        pub_loss_train[epoch] = pub_loss
        sec_loss_train[epoch] = sec_loss
        sec_acc_train[epoch] = secaccv
        e_loss_train[epoch] = e_training_loss
        d_loss_train[epoch] = d_training_loss
        KL_loss_train[epoch] = KLv
        if epoch % 10 == 3:
                pub_loss = 0
                sec_loss = 0
                e_loss = 0
                d_loss = 0
                KLv = 0
                secaccv = 0
                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    pub_tmp, sec_tmp, e_tmp, d_tmp, secacc_tmp, KL_tmp = sess.run([pub_dist, sec_dist, loss1, loss2, sec_acc, KLloss], get_feed(i, True))
                    pub_loss += pub_tmp
                    sec_loss += sec_tmp
                    e_loss += e_tmp
                    d_loss += d_tmp
                    KLv += KL_tmp
                    secaccv += secacc_tmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                d_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                secaccv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                print('Test public loss at epoch %s: %s' % (epoch, pub_loss))
                print('Test private loss at epoch %s: %s' % (epoch, sec_loss))
                pub_loss_val[epoch] = pub_loss
                sec_loss_val[epoch] = sec_loss
                sec_acc_val[epoch] = secaccv
                e_loss_train[epoch] = e_loss
                d_loss_val[epoch] = d_loss
                KL_loss_val[epoch] = KLv
                savepath = saver.save(sess, model_directory + '/celeba_privacy', global_step=epoch)
                print('Model saved at epoch %s, path is %s' % (epoch, savepath))

    np.savez(model_directory+'/celeba_trainstats_500', pub_loss_train=pub_loss_train,
                                                  sec_loss_train=sec_loss_train,
                                                  sec_acc_train=sec_acc_train,
                                                  e_loss_train=e_loss_train,
                                                  d_loss_train=d_loss_train,
                                                  KL_loss_train=KL_loss_train,
                                                  pub_loss_val=pub_loss_val,
                                                  sec_loss_val=sec_loss_val,
                                                  sec_acc_val=sec_acc_val,
                                                  e_loss_val=e_loss_val,
                                                  d_loss_val=d_loss_val,
                                                  KL_loss_val=KL_loss_val)

    sess.close()


if __name__ == "__main__":
    #list_devices()
    pc = calc_pc()
    training_privacy(pc)
