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
import components as dvib
import losses as dvibloss
# Environment variables to set GPU/CPU device
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#dvib.FLAGS.DEFINE_float("encode_coef", 1e-3, "Lambda coef for l2 term in encoding cost")
FLAGS = dvib.FLAGS
FLAGS.working_directory = "/home/rxiao"
FLAGS.summary_dir = os.path.join(os.getcwd(), 'synth')
FLAGS.dataset_size = 20000
FLAGS.test_dataset_size = 5000
FLAGS.batch_size = 500
FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
FLAGS.max_epoch = 1000
FLAGS.keep_prob = 1.0
FLAGS.learning_rate = 1e-3
FLAGS.hidden_size = 4
FLAGS.input_size = 1
FLAGS.z_size = 1
FLAGS.output_size = 1
FLAGS.private_size = 1
FLAGS.restore_model = True

eps = 1e-16
# loss1 = loss1x * encode_coef + KL/MI/sibMI
# loss2 = loss2x * decode_coef + binCE
encode_coef = 1
decode_coef = 1

def weight_variable(shape, decay_coef, stddev=0.1):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev)
  var = tf.Variable(initial)
  if decay_coef is not None:
      decay = tf.multiply(tf.nn.l2_loss(var), decay_coef, name='weight_loss')
      tf.add_to_collection('losses', decay)
  return var


def calc_pc(data=np.load(FLAGS.working_directory+"/data/synthetic/1d2gaussian.npz")):
    '''Calculate prior probabiltiy of the private variable for a 1D Gaussian
    return: vector of prior probabilities
    '''
    cs = data['c']
    stats, counts = np.unique(cs, return_counts=True)
    return [counts[1]/(counts[0]+counts[1])]

def calc_pc_weighted():
    '''Calculate prior probabiltiy of the private variable for a weighted MV Gaussian
    return: vector of prior probabilities
    '''
    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic_weighted")
    data = np.load(synth_dir + '/weightedgaussian.npz')
    cs = data['c']
    stats, counts = np.unique(cs, return_counts=True)
    return counts/cs.shape[0]

def gauss2_model(input_tensor, private_tensor, encoder="NN"):
    """ Instantiate the bernoulli-gaussian synthetic data model
    args: 
            input B x 1 tensor
            private B x 1 tensor
    output:
            xhat B x 1 tensor
            chat B x 1 tensor
            e_param_len length of encoder params
            d_param_len length of decoder params
    """
    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder", reuse=False) as scope:
                if encoder=="NN":
                    z = dvib.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                if encoder=="affine":
                    z = dvib.synth_affine_encoder(input_tensor, private_tensor, beta0, beta1)
                if encoder=="noisyaffine":
                    z = dvib.synth_affine_noisy_encoder(input_tensor, private_tensor, beta0, beta1, gamma0, gamma1)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder", reuse=False) as scope:
                xhat, chat, mean, stddev = dvib.synth_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len

    return xhat, chat, z, e_param_len, d_param_len

def gauss2_get_loss(output_tensor, private_tensor, rawc_tensor, prior_tensor, xhat, chat, z, cmetric, privmetric, order):
    """ Compute the losses of the gaussian model
        cross entropy for the private c
        L2 loss for x reconstruction
        privacy metric given and calculated as KL, MI or SibMI

        args:
                x, xhat
                c, chat
                prior
                cmetric, privmetric
                order if privmetric is given as sibson MI
    """
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean((xhat - output_tensor)**2)
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            pchat = tf.sigmoid(chat)
            sec_dist = tf.reduce_mean((pchat - private_tensor)**2)
            #correct_pred = tf.less(tf.abs(chat - private_tensor), 0.5)
            tmpchat = tf.concat([pchat, 1.0 - pchat], axis=1)
            tmppriv = tf.concat([private_tensor, 1.0 - private_tensor], axis=1)
            correct_pred = tf.equal(tf.argmax(tmpchat, axis=1), tf.argmax(tmppriv, axis=1))
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss1, KLloss = dvibloss.encoding_cost(xhat, chat, output_tensor, private_tensor, prior_tensor)
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, output_tensor, private_tensor, cmetric="CE")
    # Record losses of MI approximation and sibson MI
    h_c, h_cz, _ = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    # use alpha = 3 first, may be tuned
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, order, independent=True)
    # Distortion constraint
    lossdist = rou_tensor * tf.maximum(0.0, loss2x - D_tensor)
    # Compose losses
    if lossmetric=="KL":
        loss1 = encode_coef * lossdist + KLloss
    if lossmetric=="MI":
        loss1 = encode_coef * lossdist + I_c_cz
    if lossmetric=="sibMI":
        loss1 = encode_coef * lossdist + sibMI_c_cz
    loss2 = decode_coef * lossdist + loss2c
    #loss2 = loss2c
    #loss3 = dvibloss.get_vae_cost(mean, stddev)

    return secacc, loss2x, loss2c, KLloss, I_c_cz, sibMI_c_cz, loss1, loss2

def gauss2_get_loss_CGAP(output_tensor, private_tensor, prior_tensor, rawc_tensor, rou_tensor, D_tensor, xhat, chat, z, beta0, beta1, gamma0, gamma1, cmetric, privmetric, order):
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean((xhat - output_tensor)**2)
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            pchat = tf.sigmoid(chat)
            sec_dist = tf.reduce_mean((pchat - private_tensor)**2)
            tmpchat = tf.concat([pchat, 1.0 - pchat], axis=1)
            tmppriv = tf.concat([private_tensor, 1.0 - private_tensor], axis=1)
            correct_pred = tf.equal(tf.argmax(tmpchat, axis=1), tf.argmax(tmppriv, axis=1))
            #correct_pred = tf.less(tf.abs(chat - private_tensor), 0.5)
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #Record losses of MI approx and sibson MI
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, output_tensor, private_tensor, softmax=True, xmetric="L2")
    _, KLloss = dvibloss.encoding_cost(xhat, chat, output_tensor, private_tensor, prior_tensor, xmetric="CE", independent=False)
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, order, independent=False)
    h_c, h_cz, _ = dvibloss.MI_approx(output_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    #Calculate the loss for CGAP, using the log loss as specified
    ptilde = 0.5
    lossdist = ptilde * (beta0**2 + gamma0**2) + (1 - ptilde) * (beta1**2 + gamma1**2)
    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=private_tensor, logits=chat)) + rou_tensor * tf.maximum(0.0, lossdist - D_tensor)
    loss2 = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=private_tensor, logits=chat)) + rou_tensor * tf.maximum(0.0, lossdist - D_tensor)
    #loss = sibMI_c_cz + rou_tensor * tf.maximum(0.0, lossdist - D_tensor)
    
    return sec_acc, loss2x, loss2c, KLloss, I_c_cz, sibMI_c_cz, loss1, loss2
