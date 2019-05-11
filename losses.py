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
import components as dvibcomp
# Environment variables to set GPU/CPU device
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#dvib.FLAGS.DEFINE_float("encode_coef", 1e-3, "Lambda coef for l2 term in encoding cost")
FLAGS = dvibcomp.FLAGS
FLAGS.working_directory = "/home/rxiao"
FLAGS.summary_dir = os.path.join(os.getcwd(), 'synth')
FLAGS.dataset_size = 10000
FLAGS.test_dataset_size = 5000
FLAGS.batch_size = 250
FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
FLAGS.max_epoch = 10000
FLAGS.keep_prob = 1.0
FLAGS.learning_rate = 1e-4
FLAGS.hidden_size = 10
FLAGS.input_size = 3
FLAGS.z_size = 1
FLAGS.output_size = 1
FLAGS.private_size = 1

eps = 1e-16
# loss1 = loss1x * encode_coef + KL
# loss2 = loss2x * decode_coef + binCE
encode_coef = 1000
decode_coef = 1e-3

def get_vae_cost(mean, stddev, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean:
        stddev:
        epsilon:
    '''
    return tf.reduce_mean(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))

def get_discrim_cost(D1, D2):
    '''Cost of the discriminator
        input: D1: logits computed with a discriminator networks from real images
               D2: logits computed with a discriminator networks from generated images
        output: cross entropy loss, positive samples have label 1, negative samples 0
    '''
    return tf.reduce_mean(tf.nn.relu(D1) - D1 + tf.log(1.0 + tf.exp(-tf.abs(D1)))) + \
            tf.reduce_mean(tf.nn.relu(D2) + tf.log(1.0 + tf.exp(-tf.abs(D2))))

def get_gen_cost(D2):
    '''Loss for the generator. Maximize probability of generating images that
    discrimator cannot differentiate.

    Returns:
        see the paper
    '''
    return tf.reduce_mean(tf.nn.relu(D2) - D2 + tf.log(1.0 + tf.exp(-tf.abs(D2))))

def get_cost_x(output_tensor, target_tensor, metric="L2", softmax=True):
    '''Return the cost for X as reconstruction or predicting Y as a regular task,        if metric is CE, returns binary cross-entropy
       if metric is L2, returns L2 loss in reconstruction
    '''
    if metric=="CE":
        if softmax == False:
            cost_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_tensor, logits=target_tensor))
        else:
            cost_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_tensor, logits=target_tensor))
    if metric=="L2":
        cost_x = tf.reduce_mean((output_tensor - target_tensor)**2)
    if metric=="PPANMNIST":
        cost_x = (-1) * tf.reduce_mean(tf.multiply(target_tensor, tf.log(output_tensor))+tf.multiply(1.0-target_tensor, tf.log(1.0-output_tensor)))
    return cost_x
 
def recon_cost(output_tensor, output_tensor_priv, target_tensor, private_tensor, softmax=False, xsoftmax=True, xmetric="L2", cmetric="CE"):
    '''Cost of reconstruction
    
    output_tensor: tensor of real values
    output_tensor_priv: tensor of real values
    '''
    l2cost_x = get_cost_x(output_tensor, target_tensor, xmetric, xsoftmax)
    #L2 loss
    if cmetric=="L2":
        cost_c = tf.reduce_mean((output_tensor_priv - private_tensor)**2)
    # cross entropy loss
    # assuming p_cz is tensor of numbers in [0,1], convert to real number to take CE
    #real_logits = -log((1.0 / output_tensor_priv) - 1)
    # if p_cz are real numbers
    real_logits = output_tensor_priv
    #pdb.set_trace()
    if cmetric=="CE":
        if softmax==False:
            cost_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=private_tensor, logits=real_logits))
        else:
            cost_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=private_tensor, logits=real_logits))
    return l2cost_x, cost_c

def encoding_cost(output_tensor, output_tensor_priv, target_tensor, private_tensor, prior_tensor, xmetric="L2", independent=True):
    '''Cost of encoding x into z to maintain privacy wrt c
    
    output_tensor: tensor of real values
    output_tensor_priv: tensor of real values
    '''
    # Convert real logits of p(C|Z) into probabilities 
    if not independent:
        p_cz = tf.nn.softmax(output_tensor_priv)
    else:
        p_cz = tf.sigmoid(output_tensor_priv)
    p_c = prior_tensor
    #KL_pcz_pc = tf.reduce_mean(p_cz * tf.log(p_cz)) - tf.reduce_mean(p_cz * tf.log(p_c + eps))
    # add eps to logs to maintain stability
    KL_pcz_pc = KL_pcz_pc_approx(p_cz, p_c, independent)
    l2cost_x = get_cost_x(output_tensor, target_tensor, xmetric)
    l2cost_private = tf.reduce_mean((output_tensor_priv - private_tensor)**2)
    #loss = KL_pcz_pc
    loss = KL_pcz_pc + encode_coef * l2cost_x
    #loss = -l2cost_private + encode_coef * l2cost_x
    return loss, KL_pcz_pc

def KL_pcz_pc_approx(p_cz, p_c, independent):
    #pdb.set_trace()
    if independent:
        KL_pcz_pc = tf.reduce_mean(p_cz * tf.log(tf.div(p_cz, p_c))) + tf.reduce_mean((1. - p_cz) * tf.log(tf.div(1. - p_cz, 1. - p_c)))
    else:
        '''Here pc, pcz are vectors of prior probabilities in a discrete distribution
           wish to calculate sum_c pcz * log(pcz/pc)
        '''
        KL_pcz_pc = tf.reduce_sum(tf.reduce_mean(p_cz * tf.log(tf.maximum(p_cz / p_c, eps)), axis=0))
    return KL_pcz_pc

def MI_approx(input_tensor, private_tensor, rawc_tensor, output_tensor, output_tensor_priv, z_tensor, independent=True):
    '''Approximate the mutual information from arxiv:1802.09386
        h_c estimates the entropy by E_c E_x [E_c|x=x [-logQ_c|u]]
        h_cz estimates the conditional entropy by E_cx[E_c|x=x [-logQ_c|u]]
        l_c estimates E_z|x=x [-logQ_c|u], assuming k=1 sample of z, l_c = -logQ_c|u
    '''
    if independent:
        output_tensor_priv = tf.sigmoid(output_tensor_priv)
    else:
        output_tensor_priv = tf.nn.softmax(output_tensor_priv)
    #l_c = -tf.log(tf.clip_by_value(output_tensor_priv, eps, 1.0/eps))
    l_c = -tf.log(output_tensor_priv)
    y, idx, counts = tf.unique_with_counts(rawc_tensor)
    counts_cum = tf.cumsum(idx, exclusive=True)
    list_e_x = []
    #pdb.set_trace()
    p_c = tf.reduce_mean(private_tensor, axis=1)
    h_c = tf.reduce_sum(p_c * tf.log(1.0 / p_c))
        
    #for i in xrange(FLAGS.private_size):
    #    list_e_x.append(tf.reduce_mean(tf.gather(l_c, idx[counts_cum[i]: counts_cum[i]+counts[i]])))
    #e_x = tf.stack(list_e_x, axis=0)
    #h_c = tf.reduce_mean(e_x * tf.cast(counts, tf.float32) / FLAGS.batch_size)

    h_cz = tf.reduce_mean(l_c)
    return h_c, h_cz, l_c

def sibsonMI_approx(pz, pcz, alpha, independent=True):
    '''Approximate the sibson mutual information for tensors of pz and pc|z as I_alpha(Z;C)
       pz: array of BxN batch size x N for N dimensional z vectors sampled from pz
       pcz: array of BxM batch size x M for M dimensional c vectors given by adversary network as real values
       return a scalar tensor
    
       If pz is a tensor of z values sampled from the z distribution,
       instead we can calculate sumz by summing over the batches of z vectors
    '''
    #pdb.set_trace()
    if independent:
        pcz = tf.sigmoid(pcz)
    else:
        pcz = tf.nn.softmax(pcz)
    if independent:
        '''Here the dimensions are independent and binary, so pcz consists of probabilities
           of each dimension of c as a binary distribution
        ''' 
        sumz0 = tf.reduce_mean(tf.pow(pcz, alpha), axis=0)
        sumz1 = tf.reduce_mean(tf.pow(1.0-pcz, alpha), axis=0)
        tmp = alpha/(alpha-1.0) * tf.log(tf.pow(sumz0, 1.0/alpha) + tf.pow(sumz1, 1.0/alpha))
    else:
        '''Here the dimensions are not independent and is a vector of probabilities
        '''
        sumz = tf.reduce_mean(tf.pow(pcz, alpha), axis=0)
        tmp = alpha/(alpha-1.0) * tf.log(tf.maximum(tf.reduce_sum(tf.pow(sumz, 1.0/alpha)), eps))
    return tmp

def sibsonMI_xc_z(pz_xc, alpha):
    """ Approximate the Sibson MI for I_alpha((X,C);Z)
        pz_xc: array of BxJxN for BatchsizexJ_samples N-dimensional Z vectors 
    """
    sumxc = tf.reduce_mean(tf.pow(pz_xc, alpha), axis=2)
    sibMI = (alpha/(alpha-1.0)) * tf.log(tf.pow(sumxc, 1.0/alpha))
    return sibMI

def sibsonMI_c_z(pz, pcz, pc, alpha, independent=True):
    """ Calculate the Sibson mutual information I_alhpha(C;Z) with the given tensors
        for pz, pc, pc|z and the order alpha
        input:
            pz: tensor of BxN for N dimensional vectors sampled from pz
            pc: tensor of 1xM for M dimensional vector representing prior pc
            pcz: array of BxM batch size x M for M dimensional c vectors given by adversary network as real values
        output:
            sibMI: a scalar tensor
    """
    #pdb.set_trace()
    if independent:
        pcz = tf.sigmoid(pcz)
    else:
        pcz = tf.nn.softmax(pcz)
    if independent:
        ''' Here the dimensiosn are independent and binary, pcz is a vector of 
            probabilities of each dimension of c as a binary distribution
        '''
        sumc0 = tf.reduce_mean(tf.pow(pcz, alpha) * tf.pow(pc, 1.0 - alpha), axis=1)
        sumc1 = tf.reduce_mean(tf.pow(1.0 - pcz, alpha) * tf.pow(1.0 - pc, 1.0 - alpha), axis=1)
        sibMI = alpha/(alpha - 1.0) * tf.log(tf.reduce_mean(tf.pow(sumc0 + sumc1, 1.0/alpha)))
    else:
        ''' Here dimensions of c are dependent and is a vector of probabilities
        '''
        sumc = tf.reduce_sum(tf.pow(pcz, alpha) * tf.pow(pc, 1.0 - alpha), axis=1)
        sibMI = alpha/(alpha - 1.0) * tf.log(tf.reduce_mean(tf.pow(sumc, 1.0/alpha)))
    return sibMI
