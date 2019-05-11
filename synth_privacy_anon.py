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
import losses as dvibloss
# Environment variables to set GPU/CPU device
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#dvib.FLAGS.DEFINE_float("encode_coef", 1e-3, "Lambda coef for l2 term in encoding cost")
FLAGS = dvibcomp.FLAGS
FLAGS.working_directory = "/home/rxiao"
FLAGS.summary_dir = os.path.join(os.getcwd(), 'synth')
FLAGS.dataset_size = 10000
FLAGS.test_dataset_size = 5000
FLAGS.batch_size = 500
FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
FLAGS.max_epoch = 5000
FLAGS.keep_prob = 1.0
FLAGS.learning_rate = 1e-4
FLAGS.hidden_size = 10
FLAGS.input_size = 3
FLAGS.z_size = 1
FLAGS.output_size = 1
FLAGS.private_size = 1

eps = 1e-8
# loss1 = loss1x * encode_coef + KL
# loss2 = loss2x * decode_coef + binCE
encode_coef = 100
decode_coef = 1e-3

def calc_pc():
    '''Calculate prior probabiltiy of the private variable for a 1D Gaussian
    return: vector of prior probabilities
    '''
    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic")
    data = np.load(synth_dir + '/1d2gaussian.npz')
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

def train_2gauss(prior, lossmetric="KL"):
    '''Train model to output transformation that prevents leaking private info
    '''
    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic")
    model_directory = os.path.join(synth_dir, lossmetric+"privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])

    #load data
    data = np.load(synth_dir + '/1d2gaussian.npz')
    xs = data['x']
    cs = data['c']
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        pow_x = np.array([x, x**2, x**3]).transpose()
        x = np.array(x).reshape(FLAGS.batch_size, 1)
        c = cs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        c = np.array(c).reshape(FLAGS.batch_size, 1)
        return {input_tensor: pow_x, output_tensor: x, private_tensor: c, rawc_tensor: c.reshape(FLAGS.batch_size)}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvibcomp.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder") as scope:
                xhat, chat, mean, stddev = dvibcomp.synth_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
 
    loss1, KLloss = dvibloss.encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor)
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, output_tensor, private_tensor)
    # Experiment with alternative approximation for MI
    h_c, h_cz, l_c, e_x = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    # use alpha=3, may be tuned, calculate Sibson MI
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, 3)
    # compose losses
    if lossmetric=="KL":
        loss1 = loss1 * encode_coef + KLloss
    if lossmetric=="MI":
        loss1 = loss1 * encode_coef + I_c_cz
    if lossmetric=="sibMI":
        loss1 = loss1 * encode_coef + sibMI_c_cz
    loss2 = loss2x * decode_coef + loss2c
    loss3 = dvibloss.get_vae_cost(mean, stddev)
    #loss1 = loss1 + encode_coef * loss3
    
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean((xhat - output_tensor)**2)
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            sec_dist = tf.reduce_mean((chat - private_tensor)**2)
            correct_pred = tf.less(tf.abs(chat - private_tensor), 0.5)
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    e_train = pt.apply_optimizer(optimizer, losses=[loss1], regularize=True, include_marked=True, var_list=encode_params)
    d_train = pt.apply_optimizer(optimizer, losses=[loss2], regularize=True, include_marked=True, var_list=all_params[e_param_len:])
    # Logging matrices
    e_loss_train = np.zeros(FLAGS.max_epoch)
    d_loss_train = np.zeros(FLAGS.max_epoch)
    pub_dist_train = np.zeros(FLAGS.max_epoch)
    sec_dist_train = np.zeros(FLAGS.max_epoch)
    loss2x_train = np.zeros(FLAGS.max_epoch)
    loss2c_train = np.zeros(FLAGS.max_epoch)
    KLloss_train = np.zeros(FLAGS.max_epoch)
    MIloss_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    d_loss_val = np.zeros(FLAGS.max_epoch)
    pub_dist_val = np.zeros(FLAGS.max_epoch)
    sec_dist_val = np.zeros(FLAGS.max_epoch)
    loss2x_val = np.zeros(FLAGS.max_epoch)
    loss2c_val = np.zeros(FLAGS.max_epoch)
    KLloss_val = np.zeros(FLAGS.max_epoch)
    MIloss_val = np.zeros(FLAGS.max_epoch)
    sec_acc_val = np.zeros(FLAGS.max_epoch)
    xhat_val = []
    # Tensorboard logging
    tf.summary.scalar('e_loss', loss1)
    tf.summary.scalar('KL', KLloss)
    tf.summary.scalar('loss_x', loss2x)
    tf.summary.scalar('loss_c', loss2c)
    tf.summary.scalar('pub_dist', pub_dist)
    tf.summary.scalar('sec_dist', sec_dist)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False

    sess = tf.Session(config=config)
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
 
    for epoch in range(FLAGS.max_epoch):
        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
        pbar.start()

        pub_loss = 0
        sec_loss = 0
        sec_accv = 0
        e_training_loss = 0
        d_training_loss = 0
        KLv = 0
        MIv = 0
        loss2xv = 0
        loss2cv = 0
        #pdb.set_trace()
            
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, True)
            zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            _, d_loss_value = sess.run([d_train, loss2], feeds)
            MItmp, KLtmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([I_c_cz, KLloss, loss2x, loss2c, loss3], feeds)
            if (np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                pdb.set_trace()
                break
            #train_writer.add_summary(summary, i)
            e_training_loss += e_loss_value
            d_training_loss += d_loss_value
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            sec_accv += sec_acc_tmp
            KLv += KLtmp
            MIv += MItmp
            loss2xv += loss2xtmp
            loss2cv += loss2ctmp

        e_training_loss = e_training_loss / \
            (FLAGS.updates_per_epoch)
        d_training_loss = d_training_loss / \
            (FLAGS.updates_per_epoch)
        pub_loss /= (FLAGS.updates_per_epoch)
        sec_loss /= (FLAGS.updates_per_epoch)
        sec_accv /= (FLAGS.updates_per_epoch)
        loss2xv /= (FLAGS.updates_per_epoch)
        loss2cv /= (FLAGS.updates_per_epoch)
        KLv /= (FLAGS.updates_per_epoch)
        MIv /= (FLAGS.updates_per_epoch)

        print("Loss for E %f, and for D %f" % (e_training_loss, d_training_loss))
        print('Training public loss at epoch %s: %s' % (epoch, pub_loss))
        print('Training private loss at epoch %s: %s, private accuracy: %s' % (epoch, sec_loss, sec_accv))
        e_loss_train[epoch] = e_training_loss
        d_loss_train[epoch] = d_training_loss
        pub_dist_train[epoch] = pub_loss
        sec_dist_train[epoch] = sec_loss
        loss2x_train[epoch] = loss2xv
        loss2c_train[epoch] = loss2cv
        KLloss_train[epoch] = KLv
        MIloss_train[epoch] = MIv
        sec_acc_train[epoch] = sec_accv
        # Validation
        if epoch % 10 == 9:
                pub_loss = 0
                sec_loss = 0
                e_val_loss = 0
                d_val_loss = 0
                loss2xv = 0
                loss2cv = 0
                KLv = 0
                MIv = 0
                sec_accv = 0

                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    feeds = get_feed(i, False)
                    pub_loss += sess.run(pub_dist, feeds)
                    sec_loss += sess.run(sec_dist, feeds)
                    e_val_loss += sess.run(loss1, feeds)
                    d_val_loss += sess.run(loss2, feeds)
                    MItmp, KLtmp, loss2xtmp, loss2ctmp, sec_acc_tmp = sess.run([I_c_cz, KLloss, loss2x, loss2c, sec_acc], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        xhat_val.extend(sess.run(xhat, feeds))
                    #test_writer.add_summary(summary, i)
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    MIv += MItmp
                    loss2xv += loss2xtmp
                    loss2cv += loss2ctmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                d_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)

                print('Test public loss at epoch %s: %s' % (epoch, pub_loss))
                print('Test private loss at epoch %s: %s' % (epoch, sec_loss))
                e_loss_val[epoch] = e_val_loss
                d_loss_val[epoch] = d_val_loss
                pub_dist_val[epoch] = pub_loss
                sec_dist_val[epoch] = sec_loss
                loss2x_val[epoch] = loss2xv
                loss2c_val[epoch] = loss2cv
                KLloss_val[epoch] = KLv
                MIloss_val[epoch] = MIv
                sec_acc_val[epoch] = sec_accv
 
                if not(np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                    savepath = saver.save(sess, model_directory + '/synth_privacy', global_step=epoch)
                    print('Model saved at epoch %s, path is %s' % (epoch, savepath))

    np.savez(os.path.join(model_directory, 'synth_trainstats'), e_loss_train=e_loss_train,
                                                  d_loss_train=d_loss_train,
                                                  pub_dist_train=pub_dist_train,
                                                  sec_dist_train=sec_dist_train,
                                                  loss2x_train = loss2x_train,
                                                  loss2c_train = loss2c_train,
                                                  KLloss_train = KLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  d_loss_val=d_loss_val,
                                                  pub_dist_val=pub_dist_val,
                                                  sec_dist_val=sec_dist_val,
                                                  loss2x_val = loss2x_val,
                                                  loss2c_val = loss2c_val,
                                                  KLloss_val = KLloss_val,
                                                  MIloss_val = MIloss_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

    sess.close()
    return

def train_gauss_discrim(prior):
    '''Train model to output transformation that prevents leaking private info, with weighted vector input data
    input: prior [1xM] probabilities of each class label in the dataset
    '''
    FLAGS.dataset_size = 10000
    FLAGS.test_dataset_size = 5000
    FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
    FLAGS.input_size=10
    FLAGS.z_size=40
    FLAGS.output_size=10
    FLAGS.private_size=10
    FLAGS.hidden_size=100

    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic_weighted")
    # Change model directory for logging purposes
    model_directory = os.path.join(synth_dir, "discrim_MI_privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])

    #load data
    data = np.load(synth_dir + '/weightedgaussian.npz')
    xs = data['x']
    cs = data['c']
    #convert class labels to one hot encoding
    onehot_cs = np.eye(np.max(cs)+1)[cs]
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no : offset + FLAGS.batch_size * (batch_no + 1)]
        onehot_c = onehot_cs[offset + FLAGS.batch_size * batch_no : offset + FLAGS.batch_size * (batch_no + 1)]
        c = cs[offset + FLAGS.batch_size * batch_no : offset + FLAGS.batch_size * (batch_no + 1)]
        #if x.shape==(0, 10):
        #    pdb.set_trace()
        return {input_tensor: x, output_tensor: x, private_tensor: onehot_c, rawc_tensor: c}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvibcomp.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder") as scope:
                xhat, chat, mean, stddev = dvibcomp.synth_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
            with tf.variable_scope("discrim") as scope:
                D1 = dvibcomp.synth_discriminator(input_tensor) # positive samples
            with tf.variable_scope("discrim", reuse=True) as scope:
                D2 = dvibcomp.synth_discriminator(xhat) # negative samples
                all_params = tf.trainable_variables()
                discrim_len = len(all_params) - (d_param_len + e_param_len)
    
    #Calculate losses
    _, KLloss = dvibloss.encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor)
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, output_tensor, private_tensor, softmax=True)
    # Experiment with alternative approximation for MI
    h_c, h_cz, l_c, e_x = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    
    loss_g = dvibloss.get_gen_cost(D2)
    loss_d = dvibloss.get_discrim_cost(D1, D2)
    loss1 = loss_g * encode_coef + I_c_cz
    loss2 = loss_g * decode_coef + loss2c
    loss_vae = dvibloss.get_vae_cost(mean, stddev)
 
    
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean((xhat - output_tensor)**2)
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            sec_dist = tf.reduce_mean((chat - private_tensor)**2)
            correct_pred = tf.equal(tf.argmax(chat, axis=1), tf.argmax(private_tensor, axis=1))
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    e_train = pt.apply_optimizer(optimizer, losses=[loss1], regularize=True, include_marked=True, var_list=encode_params)
    g_train = pt.apply_optimizer(optimizer, losses=[loss2], regularize=True, include_marked=True, var_list=all_params[e_param_len:]) # generator/decoder training op
    d_train = pt.apply_optimizer(optimizer, losses=[loss_d], regularize=True, include_marked=True, var_list=all_params[e_param_len+d_param_len:])
    # Logging matrices
    e_loss_train = np.zeros(FLAGS.max_epoch)
    g_loss_train = np.zeros(FLAGS.max_epoch)
    d_loss_train = np.zeros(FLAGS.max_epoch)
    vae_loss_train = np.zeros(FLAGS.max_epoch)
    pub_dist_train = np.zeros(FLAGS.max_epoch)
    sec_dist_train = np.zeros(FLAGS.max_epoch)
    loss2x_train = np.zeros(FLAGS.max_epoch)
    loss2c_train = np.zeros(FLAGS.max_epoch)
    KLloss_train = np.zeros(FLAGS.max_epoch)
    MIloss_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    g_loss_val = np.zeros(FLAGS.max_epoch)
    d_loss_val = np.zeros(FLAGS.max_epoch)
    vae_loss_val = np.zeros(FLAGS.max_epoch)
    pub_dist_val = np.zeros(FLAGS.max_epoch)
    sec_dist_val = np.zeros(FLAGS.max_epoch)
    loss2x_val = np.zeros(FLAGS.max_epoch)
    loss2c_val = np.zeros(FLAGS.max_epoch)
    KLloss_val = np.zeros(FLAGS.max_epoch)
    MIloss_val = np.zeros(FLAGS.max_epoch)
    sec_acc_val = np.zeros(FLAGS.max_epoch)
    xhat_val = []
    # Tensorboard logging
    #tf.summary.scalar('e_loss', loss_g)
    #tf.summary.scalar('KL', KLloss)
    #tf.summary.scalar('loss_x', loss2x)
    #tf.summary.scalar('loss_c', loss2c)
    #tf.summary.scalar('pub_dist', pub_dist)
    #tf.summary.scalar('sec_dist', sec_dist)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False

    sess = tf.Session(config=config)
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
 
    for epoch in range(FLAGS.max_epoch):
        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
        pbar.start()

        pub_loss = 0
        sec_loss = 0
        sec_accv = 0
        e_training_loss = 0
        g_training_loss = 0
        d_training_loss = 0
        KLv = 0
        MIv = 0
        loss2xv = 0
        loss2cv = 0
        loss3v = 0
        #pdb.set_trace()
            
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, True)
            zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
            I_c_czv, h_cv, h_czv, l_cv, e_xv = sess.run([I_c_cz, h_c, h_cz, l_c, e_x], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            _, g_loss_value = sess.run([g_train, loss2], feeds)
            _, d_loss_value = sess.run([d_train, loss_d], feeds)
            KLtmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([KLloss, loss2x, loss2c, loss_vae], feeds)
            if (np.isnan(e_loss_value) or np.isnan(g_loss_value) or np.isnan(d_loss_value)):
                pdb.set_trace()
                break
            #train_writer.add_summary(summary, i)
            e_training_loss += e_loss_value
            g_training_loss += g_loss_value
            d_training_loss += d_loss_value
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            sec_accv += sec_acc_tmp
            KLv += KLtmp
            MIv += I_c_czv
            loss2xv += loss2xtmp
            loss2cv += loss2ctmp
            loss3v += loss2ctmp

        e_training_loss = e_training_loss / \
            (FLAGS.updates_per_epoch)
        g_training_loss = g_training_loss / \
            (FLAGS.updates_per_epoch)
        d_training_loss = d_training_loss / \
            (FLAGS.updates_per_epoch)
        pub_loss /= (FLAGS.updates_per_epoch)
        sec_loss /= (FLAGS.updates_per_epoch)
        sec_accv /= (FLAGS.updates_per_epoch)
        loss2xv /= (FLAGS.updates_per_epoch)
        loss2cv /= (FLAGS.updates_per_epoch)
        loss3v /= (FLAGS.updates_per_epoch)
        KLv /= (FLAGS.updates_per_epoch)

        print("Loss for E %f, for G %f, for D %f" % (e_training_loss, g_training_loss, d_training_loss))
        print('Training public loss at epoch %s: %s' % (epoch, pub_loss))
        print('Training private loss at epoch %s: %s, private accuracy: %s' % (epoch, sec_loss, sec_accv))
        e_loss_train[epoch] = e_training_loss
        g_loss_train[epoch] = g_training_loss
        d_loss_train[epoch] = d_training_loss
        pub_dist_train[epoch] = pub_loss
        sec_dist_train[epoch] = sec_loss
        loss2x_train[epoch] = loss2xv
        loss2c_train[epoch] = loss2cv
        vae_loss_train[epoch] = loss3v
        KLloss_train[epoch] = KLv
        MIloss_train[epoch] = MIv
        sec_acc_train[epoch] = sec_accv
        # Validation
        if epoch % 10 == 9:
                pub_loss = 0
                sec_loss = 0
                e_val_loss = 0
                g_val_loss = 0
                d_val_loss = 0
                loss2xv = 0
                loss2cv = 0
                loss3v = 0
                KLv = 0
                MIv = 0
                sec_accv = 0

                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    feeds = get_feed(i, False)
                    pub_loss += sess.run(pub_dist, feeds)
                    sec_loss += sess.run(sec_dist, feeds)
                    e_val_loss += sess.run(loss1, feeds)
                    g_val_loss += sess.run(loss2, feeds)
                    d_val_loss += sess.run(loss_d, feeds)
                    KLtmp, loss2xtmp, loss2ctmp, sec_acc_tmp, loss3tmp = sess.run([KLloss, loss2x, loss2c, sec_acc, loss_vae], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        xhat_val.extend(sess.run(xhat, feeds))
                    #test_writer.add_summary(summary, i)
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    MIv += I_c_czv
                    loss2xv += loss2xtmp
                    loss2cv += loss2ctmp
                    loss3v += loss3tmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                g_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                d_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss3v /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)

                print('Test public loss at epoch %s: %s' % (epoch, pub_loss))
                print('Test private loss at epoch %s: %s' % (epoch, sec_loss))
                e_loss_val[epoch] = e_val_loss
                g_loss_val[epoch] = g_val_loss
                d_loss_val[epoch] = d_val_loss
                pub_dist_val[epoch] = pub_loss
                sec_dist_val[epoch] = sec_loss
                loss2x_val[epoch] = loss2xv
                loss2c_val[epoch] = loss2cv
                vae_loss_val[epoch] = loss3v
                KLloss_val[epoch] = KLv
                MIloss_val[epoch] = MIv
                sec_acc_val[epoch] = sec_accv
 
                if not(np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                    savepath = saver.save(sess, model_directory + '/synth_privacy', global_step=epoch)
                    print('Model saved at epoch %s, path is %s' % (epoch, savepath))

    np.savez(os.path.join(model_directory, 'synth_trainstats'), e_loss_train=e_loss_train,
                                                  g_loss_train=g_loss_train,
                                                  d_loss_train=d_loss_train,
                                                  pub_dist_train=pub_dist_train,
                                                  sec_dist_train=sec_dist_train,
                                                  loss2x_train = loss2x_train,
                                                  loss2c_train = loss2c_train,
                                                  vae_loss_train = vae_loss_train,
                                                  KLloss_train = KLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  g_loss_val=g_loss_val,
                                                  d_loss_val=d_loss_val,
                                                  pub_dist_val=pub_dist_val,
                                                  sec_dist_val=sec_dist_val,
                                                  loss2x_val = loss2x_val,
                                                  loss2c_val = loss2c_val,
                                                  vae_loss_val = vae_loss_val,
                                                  KLloss_val = KLloss_val,
                                                  MIloss_val = MIloss_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

   

    sess.close()



if __name__ == '__main__':
    prior = calc_pc()
    train_2gauss(prior, lossmetric="MI")
    #pc = calc_pc_weighted()
    #train_gauss_discrim(pc)

