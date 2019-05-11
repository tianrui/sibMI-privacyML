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

def train_2gauss(prior, lossmetric="KL", order=20, K_iters=50, D=3):
    '''Train model to output transformation that prevents leaking private info
    '''
    # Set a secret random seed for all models
    tf.set_random_seed(515319)


    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic")
    model_directory = os.path.join(synth_dir, lossmetric+"_"+"privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef)+"_D"+str(D)+"_order"+str(order))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])
    rou_tensor = tf.placeholder(tf.float32)
    D_tensor = tf.placeholder(tf.float32)

    #load data, if ends in 3_m3 then means are [3, -3]
    data = np.load(synth_dir + '/1d2gaussian_3_m3.npz')
    xs = data['x']
    cs = data['c']
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        pow_x = np.array([x, x**2, x**3]).transpose()
        x = np.array(x).reshape(FLAGS.batch_size, 1)
        c = cs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        c = np.array(c).reshape(FLAGS.batch_size, 1)
        rawc = c.reshape(FLAGS.batch_size)
        return {input_tensor: x, output_tensor: x, private_tensor: c, rawc_tensor: rawc}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder", reuse=False) as scope:
                z = dvib.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder", reuse=False) as scope:
                xhat, chat, mean, stddev = dvib.synth_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
  
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

    loss1, KLloss = dvibloss.encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor)
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
    loss3 = dvibloss.get_vae_cost(mean, stddev)

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
    npKLloss_train = np.zeros(FLAGS.max_epoch)
    MIloss_train = np.zeros(FLAGS.max_epoch)
    sibMIloss_train = np.zeros(FLAGS.max_epoch)
    npsibMIloss_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    d_loss_val = np.zeros(FLAGS.max_epoch)
    pub_dist_val = np.zeros(FLAGS.max_epoch)
    sec_dist_val = np.zeros(FLAGS.max_epoch)
    loss2x_val = np.zeros(FLAGS.max_epoch)
    loss2c_val = np.zeros(FLAGS.max_epoch)
    KLloss_val = np.zeros(FLAGS.max_epoch)
    npKLloss_val = np.zeros(FLAGS.max_epoch)
    MIloss_val = np.zeros(FLAGS.max_epoch)
    sibMIloss_val = np.zeros(FLAGS.max_epoch)
    npsibMIloss_val = np.zeros(FLAGS.max_epoch)
    sec_acc_val = np.zeros(FLAGS.max_epoch)
    xhat_val = []
    # Tensorboard logging
    tf.summary.scalar('e_loss', loss1)
    tf.summary.scalar('KL', KLloss)
    tf.summary.scalar('loss_x', loss2x)
    tf.summary.scalar('loss_c', loss2c)
    tf.summary.scalar('pub_dist', pub_dist)
    tf.summary.scalar('sec_dist', sec_dist)
    # Rou tensor values, penalty parameter for the distortion constraint
    rou_values = np.linspace(0, 1000, FLAGS.max_epoch)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False
    sess = tf.Session(config=config)

    #Attempt to restart from last checkpt
    checkpt = tf.train.latest_checkpoint(model_directory)
    if checkpt != None and FLAGS.restore_model==True:
        saver.restore(sess, checkpt)
        print("Restored model from checkpoint %s" % (checkpt))
    else:
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
        npKLv = 0
        MIv = 0
        sibMIv = 0
        npsibMIv = 0
        loss2xv = 0
        loss2cv = 0
        #pdb.set_trace()
        if epoch == FLAGS.max_epoch-1:
            pdb.set_trace()
            
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            # Feed values of data and rou, D parameters
            feeds = get_feed(i, True)
            feeds[rou_tensor] = rou_values[epoch]
            feeds[D_tensor] = D
            zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            for j in xrange(K_iters):
                _, d_loss_value = sess.run([d_train, loss2], feeds)
            MItmp, sibMItmp, KLtmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([I_c_cz, sibMI_c_cz, KLloss, loss2x, loss2c, loss3], feeds)
            pchatv = 1.0 / (1.0 + np.exp(-chatv))
            npKLtmp = np.average(pchatv * np.log(np.clip(pchatv/prior[0], eps, None))) + np.average((1.0-pchatv) * np.log(np.clip((1.0-pchatv)/(1.0-prior[0]), eps, None)))
            npsibMItmp = (order/(order-1.0)) * np.log(np.clip(np.average(pchatv**order)**(1.0/order) + np.average((1.0-pchatv)**order)**(1.0/order), eps, None))
 
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
            npKLv += npKLtmp
            MIv += MItmp
            sibMIv += npsibMItmp
            npsibMIv += sibMItmp
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
        npKLv /= (FLAGS.updates_per_epoch)
        MIv /= (FLAGS.updates_per_epoch)
        sibMIv /= (FLAGS.updates_per_epoch)
        npsibMIv /= (FLAGS.updates_per_epoch)

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
        npKLloss_train[epoch] = npKLv
        sibMIloss_train[epoch] = sibMIv
        npsibMIloss_train[epoch] = npsibMIv
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
                sibMIv = 0
                sec_accv = 0

                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    feeds = get_feed(i, False)
                    feeds[rou_tensor] = rou_values[epoch]
                    feeds[D_tensor] = D
                    pub_loss += sess.run(pub_dist, feeds)
                    sec_loss += sess.run(sec_dist, feeds)
                    e_val_loss += sess.run(loss1, feeds)
                    d_val_loss += sess.run(loss2, feeds)
                    MItmp, sibMItmp, KLtmp, loss2xtmp, loss2ctmp, sec_acc_tmp = sess.run([I_c_cz, sibMI_c_cz, KLloss, loss2x, loss2c, sec_acc], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        xhat_val.extend(sess.run(xhat, feeds))
                    #test_writer.add_summary(summary, i)
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    MIv += MItmp
                    sibMIv += sibMItmp
                    loss2xv += loss2xtmp
                    loss2cv += loss2ctmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                d_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                MIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sibMIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)

                print('Test public loss at epoch %s: %s' % (epoch, pub_loss))
                print('Test private loss at epoch %s: %s' % (epoch, sec_loss))
                print('Test private accuracy at epoch %s: %s' % (epoch, sec_accv))
                e_loss_val[epoch] = e_val_loss
                d_loss_val[epoch] = d_val_loss
                pub_dist_val[epoch] = pub_loss
                sec_dist_val[epoch] = sec_loss
                loss2x_val[epoch] = loss2xv
                loss2c_val[epoch] = loss2cv
                KLloss_val[epoch] = KLv
                MIloss_val[epoch] = MIv
                sibMIloss_val[epoch] = sibMIv
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
                                                  npKLloss_train = npKLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sibMIloss_train = sibMIloss_train,
                                                  npsibMIloss_train = npsibMIloss_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  d_loss_val=d_loss_val,
                                                  pub_dist_val=pub_dist_val,
                                                  sec_dist_val=sec_dist_val,
                                                  loss2x_val = loss2x_val,
                                                  loss2c_val = loss2c_val,
                                                  KLloss_val = KLloss_val,
                                                  MIloss_val = MIloss_val,
                                                  sibMIloss_val = sibMIloss_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

    sess.close()
    return

def train_2gauss_compare_theoretic(prior, lossmetric="KL", order=20, K_iters=5, D=7, I=0):
    '''Train model to output transformation that prevents leaking private info
        compare with theoretic framework with the approximate upper bound on Sibson MI
    '''
    # Set a secret random seed for all models
    tf.set_random_seed(515319)


    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic")
    save_dir = os.path.join(synth_dir, "compare_theoretic/"+lossmetric+"/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_directory = os.path.join(save_dir, lossmetric+"_"+"privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef)+"_D"+str(D)+"_order"+str(order)+"_I"+str(I))
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])
    rou_tensor = tf.placeholder(tf.float32)
    D_tensor = tf.placeholder(tf.float32)
    # Initialize vectors for class dependent affine transformation
    #pdb.set_trace()
    #beta0 = weight_variable([1, FLAGS.z_size], None)
    #beta1 = weight_variable([1, FLAGS.z_size], None)
    #gamma0 = weight_variable([1, FLAGS.z_size], None)
    #gamma1 = weight_variable([1, FLAGS.z_size], None)
    beta0 = tf.Variable(tf.zeros([1,FLAGS.z_size]))
    beta1 = tf.Variable(tf.zeros([1,FLAGS.z_size]))
    #gamma0 = tf.Variable(tf.zeros([1,FLAGS.z_size]))
    #gamma1 = tf.Variable(tf.zeros([1,FLAGS.z_size]))
    gamma = tf.Variable(tf.zeros([1,FLAGS.z_size]))



    #load data, if ends in _3_m3 the means are [3, -3]
    data = np.load(synth_dir + '/1d2gaussian_3_m3.npz')
    xs = data['x']
    cs = data['c']
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        pow_x = np.array([x, x**2, x**3]).transpose()
        x = np.array(x).reshape(FLAGS.batch_size, 1)
        c = cs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        c = np.array(c).reshape(FLAGS.batch_size, 1)
        rawc = c.reshape(FLAGS.batch_size)
        return {input_tensor: x, output_tensor: x, private_tensor: c, rawc_tensor: rawc}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder", reuse=False) as scope:
                #z = dvib.synth_affine_noisy_encoder(input_tensor, private_tensor, beta0, beta1, gamma0, gamma1)
                #z = dvib.synth_affine_indepnoisy_encoder(input_tensor, private_tensor, beta0, beta1, gamma)
                #z = dvib.synth_affine_encoder(input_tensor, private_tensor, beta0, beta1)
                z = dvib.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder", reuse=False) as scope:
                xhat, chat, mean, stddev = dvib.synth_predictor(z, sampling=False) # use sampling true when z tensor is distribution parameters
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
  
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

    _, KLloss = dvibloss.encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor, xmetric="L2", independent=True)
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, output_tensor, private_tensor, cmetric="CE")
    # Record losses of MI approximation and sibson MI
    h_c, h_cz, _ = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    # use alpha = 3 first, may be tuned
    # Calculate sibson I(Z;C)
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, order, independent=True)
    # Calculate sibson I(C;Z)
    sibMI_c_z = dvibloss.sibsonMI_c_z(z, chat, prior_tensor, order, independent=True)
    # Distortion constraint
    ptilde = 0.5
    lossparams = tf.reduce_mean(ptilde * (beta0**2) + (1-ptilde) * (beta1**2))
    #lossparams = tf.reduce_mean(ptilde * (beta0**2) + (1-ptilde) * (beta1**2) + gamma**2)
    #lossparams = tf.reduce_mean(ptilde * (beta0**2 + gamma0**2) + (1-ptilde) * (beta1**2 + gamma1**2))
    #lossdist = rou_tensor * tf.maximum(0.0, lossparams - D_tensor)
    #lossdist = rou_tensor * tf.maximum(0.0, lossparams - D_tensor) + loss2x
    #lossdist = rou_tensor * tf.maximum(0.0, lossparams - D_tensor)
    lossdist = rou_tensor * tf.maximum(0.0, loss2x - D_tensor)
    # Compose losses
    if lossmetric=="KL":
        loss1 = encode_coef * lossdist + KLloss
    if lossmetric=="MI":
        loss1 = encode_coef * lossdist + I_c_cz
    if lossmetric=="sibMI":
        if I > 0:
            #loss1 = encode_coef * lossdist + 10 * tf.maximum(sibMI_c_cz - I, 0.0)
            loss1 = encode_coef * lossdist + 1 * tf.maximum(sibMI_c_z - I, 0.0)
            #loss1 = encode_coef * lossdist + tf.maximum(sibMI_c_cz - I, 0.0)
        else:
            loss1 = encode_coef * lossdist + 10 * sibMI_c_cz
            #loss1 = encode_coef * lossparam + 1 * sibMI_c_z
            #loss1 = encode_coef * lossdist + sibMI_c_cz
    loss2 = decode_coef * lossdist + loss2c
    #loss2 = decode_coef * lossdist - sibMI_c_cz
    #loss2 = loss2c
    loss3 = dvibloss.get_vae_cost(mean, stddev)

    e_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    d_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    e_train = pt.apply_optimizer(e_optimizer, losses=[loss1], regularize=True, include_marked=True, var_list=encode_params)
    d_train = pt.apply_optimizer(d_optimizer, losses=[loss2], regularize=True, include_marked=True, var_list=all_params[e_param_len:])
    # Logging matrices
    e_loss_train = np.zeros(FLAGS.max_epoch)
    d_loss_train = np.zeros(FLAGS.max_epoch)
    pub_dist_train = np.zeros(FLAGS.max_epoch)
    sec_dist_train = np.zeros(FLAGS.max_epoch)
    loss2x_train = np.zeros(FLAGS.max_epoch)
    loss2c_train = np.zeros(FLAGS.max_epoch)
    lossparam_train = np.zeros(FLAGS.max_epoch)
    KLloss_train = np.zeros(FLAGS.max_epoch)
    npKLloss_train = np.zeros(FLAGS.max_epoch)
    MIloss_train = np.zeros(FLAGS.max_epoch)
    sibMIloss_train = np.zeros(FLAGS.max_epoch)
    sibMI_c_z_train = np.zeros(FLAGS.max_epoch)
    npsibMIloss_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    d_loss_val = np.zeros(FLAGS.max_epoch)
    pub_dist_val = np.zeros(FLAGS.max_epoch)
    sec_dist_val = np.zeros(FLAGS.max_epoch)
    loss2x_val = np.zeros(FLAGS.max_epoch)
    loss2c_val = np.zeros(FLAGS.max_epoch)
    lossparam_val = np.zeros(FLAGS.max_epoch)
    KLloss_val = np.zeros(FLAGS.max_epoch)
    npKLloss_val = np.zeros(FLAGS.max_epoch)
    MIloss_val = np.zeros(FLAGS.max_epoch)
    sibMIloss_val = np.zeros(FLAGS.max_epoch)
    sibMI_c_z_val = np.zeros(FLAGS.max_epoch)
    npsibMIloss_val = np.zeros(FLAGS.max_epoch)
    sec_acc_val = np.zeros(FLAGS.max_epoch)
    xhat_val = []
    # Tensorboard logging
    tf.summary.scalar('e_loss', loss1)
    tf.summary.scalar('KL', KLloss)
    tf.summary.scalar('loss_x', loss2x)
    tf.summary.scalar('loss_c', loss2c)
    tf.summary.scalar('pub_dist', pub_dist)
    tf.summary.scalar('sec_dist', sec_dist)
    # Rou tensor values, penalty parameter for the distortion constraint
    rou_values = np.linspace(0, 2.0, FLAGS.max_epoch)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False
    sess = tf.Session(config=config)
    
    #Attempt to restart from last checkpt
    if FLAGS.restore_model==True:
        if os.path.exists(model_directory):
            checkpt = tf.train.latest_checkpoint(model_directory)
        else:
            checkpt = None
        if checkpt != None:
            saver.restore(sess, checkpt)
            print("Restored model from checkpoint %s" % (checkpt))
        else:
            sess.run(init)
    else:
        sess.run(init)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
 
    #pdb.set_trace()
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
        npKLv = 0
        MIv = 0
        sibMIv = 0
        sibMI_c_zv = 0
        npsibMIv = 0
        loss2xv = 0
        loss2cv = 0
        lossparamv = 0
        #pdb.set_trace()
        if epoch == FLAGS.max_epoch-1:
            pdb.set_trace()
            #print(sess.run([beta0,beta1,gamma0,gamma1]))
            print(sess.run([beta0,beta1,gamma]))
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            # Feed values of data and rou, D parameters
            feeds = get_feed(i, True)
            feeds[rou_tensor] = rou_values[epoch]
            feeds[D_tensor] = D
            zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
            MItmp, sibMItmp, sibMI_c_ztmp, KLtmp, loss2xtmp, loss2ctmp, lossparamtmp = sess.run([I_c_cz, sibMI_c_cz, sibMI_c_z, KLloss, loss2x, loss2c, lossparams], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            for i in xrange(K_iters):
                _, d_loss_value = sess.run([d_train, loss2], feeds)
            pchatv = 1.0 / (1.0 + np.exp(np.clip(-chatv, -500, 500)))
            npKLtmp = np.average(pchatv * np.log(np.clip(pchatv/prior[0], eps, None))) + np.average((1.0-pchatv) * np.log(np.clip((1.0-pchatv)/(1.0-prior[0]), eps, None)))
            npsibMItmp = (order/(order-1.0)) * np.log(np.clip(np.average(pchatv**order)**(1.0/order) + np.average((1.0-pchatv)**order)**(1.0/order), eps, None))
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
            npKLv += npKLtmp
            MIv += MItmp
            sibMIv += sibMItmp
            sibMI_c_zv += sibMI_c_ztmp
            npsibMIv += sibMItmp
            loss2xv += loss2xtmp
            loss2cv += loss2ctmp
            lossparamv += lossparamtmp

        e_training_loss = e_training_loss / \
            (FLAGS.updates_per_epoch)
        d_training_loss = d_training_loss / \
            (FLAGS.updates_per_epoch)
        pub_loss /= (FLAGS.updates_per_epoch)
        sec_loss /= (FLAGS.updates_per_epoch)
        sec_accv /= (FLAGS.updates_per_epoch)
        loss2xv /= (FLAGS.updates_per_epoch)
        loss2cv /= (FLAGS.updates_per_epoch)
        lossparamv /= (FLAGS.updates_per_epoch)
        KLv /= (FLAGS.updates_per_epoch)
        npKLv /= (FLAGS.updates_per_epoch)
        MIv /= (FLAGS.updates_per_epoch)
        sibMIv /= (FLAGS.updates_per_epoch)
        sibMI_c_zv /= (FLAGS.updates_per_epoch)
        npsibMIv /= (FLAGS.updates_per_epoch)

        print("Loss for E %f, and for D %f" % (e_training_loss, d_training_loss))
        #print('Training public loss at epoch %s: %s' % (epoch, pub_loss))
        print('at epoch %s, training CE loss: %s, private accuracy: %s, loss2x: %s, KL: %s, sibMI_c_z: %s' % (epoch, loss2cv, sec_accv, loss2xv, KLv, sibMI_c_zv))
        e_loss_train[epoch] = e_training_loss
        d_loss_train[epoch] = d_training_loss
        pub_dist_train[epoch] = pub_loss
        sec_dist_train[epoch] = sec_loss
        loss2x_train[epoch] = loss2xv
        loss2c_train[epoch] = loss2cv
        lossparam_train[epoch] = lossparamv
        KLloss_train[epoch] = KLv
        npKLloss_train[epoch] = npKLv
        sibMIloss_train[epoch] = sibMIv
        sibMI_c_z_train[epoch] = sibMI_c_zv
        npsibMIloss_train[epoch] = npsibMIv
        sec_acc_train[epoch] = sec_accv
        # Validation
        if epoch % 10 == 9:
            #pdb.set_trace()
            beta0v, beta1v, lossparamv = sess.run([beta0,beta1,lossparams])
            print(beta0v, beta1v, lossparamv)
            pub_loss = 0
            sec_loss = 0
            e_val_loss = 0
            d_val_loss = 0
            loss2xv = 0
            loss2cv = 0
            lossparamv = 0
            KLv = 0
            MIv = 0
            sibMIv = 0
            sibMI_c_zv = 0
            sec_accv = 0

            for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                feeds = get_feed(i, False)
                feeds[rou_tensor] = rou_values[epoch]
                feeds[D_tensor] = D
                pub_loss += sess.run(pub_dist, feeds)
                sec_loss += sess.run(sec_dist, feeds)
                e_val_loss += sess.run(loss1, feeds)
                d_val_loss += sess.run(loss2, feeds)
                zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
                MItmp, sibMItmp, sibMI_c_ztmp, KLtmp, loss2xtmp, loss2ctmp, lossparamtmp, sec_acc_tmp = sess.run([I_c_cz, sibMI_c_cz, sibMI_c_z, KLloss, loss2x, loss2c, lossparams, sec_acc], feeds)
                if (epoch >= FLAGS.max_epoch - 10):
                    xhat_val.extend(sess.run(xhat, feeds))
                #test_writer.add_summary(summary, i)
                sec_accv += sec_acc_tmp
                KLv += KLtmp
                MIv += MItmp
                sibMIv += sibMItmp
                sibMI_c_zv += sibMI_c_ztmp
                loss2xv += loss2xtmp
                loss2cv += loss2ctmp
                lossparamv += lossparamtmp

            pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            d_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            lossparamv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            MIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            sibMIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            sibMI_c_zv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
            sec_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)

            print('Test public loss at epoch %s: %s' % (epoch, pub_loss))
            print('Test private loss at epoch %s: %s' % (epoch, sec_loss))
            print('Test private accuracy at epoch %s: %s' % (epoch, sec_accv))
            e_loss_val[epoch] = e_val_loss
            d_loss_val[epoch] = d_val_loss
            pub_dist_val[epoch] = pub_loss
            sec_dist_val[epoch] = sec_loss
            loss2x_val[epoch] = loss2xv
            loss2c_val[epoch] = loss2cv
            lossparam_val[epoch] = lossparamv
            KLloss_val[epoch] = KLv
            MIloss_val[epoch] = MIv
            sibMIloss_val[epoch] = sibMIv
            sibMI_c_z_val[epoch] = sibMI_c_zv
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
                                                  lossparam_train = lossparam_train,
                                                  KLloss_train = KLloss_train,
                                                  npKLloss_train = npKLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sibMIloss_train = sibMIloss_train,
                                                  sibMI_c_z_train = sibMI_c_z_train,
                                                  npsibMIloss_train = npsibMIloss_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  d_loss_val=d_loss_val,
                                                  pub_dist_val=pub_dist_val,
                                                  sec_dist_val=sec_dist_val,
                                                  loss2x_val = loss2x_val,
                                                  loss2c_val = loss2c_val,
                                                  lossparam_val = lossparam_val,
                                                  KLloss_val = KLloss_val,
                                                  MIloss_val = MIloss_val,
                                                  sibMIloss_val = sibMIloss_val,
                                                  sibMI_c_z_val = sibMI_c_z_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

    sess.close()
    return

def train_gauss_weighted(prior):
    '''Train model to output transformation that prevents leaking private info, with weighted vector input data
    input: prior [1xM] probabilities of each class label in the dataset
    '''
    # Set a secret random seed for all models
    tf.set_random_seed(515319)


    FLAGS.dataset_size = 10000
    FLAGS.test_dataset_size = 5000
    FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
    FLAGS.input_size=10
    FLAGS.z_size=10
    FLAGS.output_size=10
    FLAGS.private_size=10
    FLAGS.hidden_size=20

    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic_weighted")
    model_directory = os.path.join(synth_dir, "weighted_privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])

    #load data
    data = np.load(synth_dir + '/weightedgaussian.npz')
    xs = data['x']
    cs = data['c']
    #convert class labels to one hot encoding
    cs = np.eye(np.max(cs)+1)[cs]
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        c = cs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        return {input_tensor: x, output_tensor: x, private_tensor: c}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvib.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder") as scope:
                xhat, chat, mean, stddev = dvib.synth_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
    
    #Calculate losses
    _, KLloss = encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor)
    loss2x, loss2c = recon_cost(xhat, chat, output_tensor, private_tensor)
    loss_g = loss2x
    loss1 = loss_g * encode_coef + KLloss
    loss2 = loss_g * decode_coef + loss2c
    loss_vae = get_vae_cost(mean, stddev)
    #loss1 = loss1 + encode_coef * loss3
    
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
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    e_train = pt.apply_optimizer(optimizer, losses=[loss1], regularize=True, include_marked=True, var_list=encode_params) # encoder training op
    g_train = pt.apply_optimizer(optimizer, losses=[loss2], regularize=True, include_marked=True, var_list=all_params[e_param_len:]) # generator/decoder training op
    # Logging matrices
    e_loss_train = np.zeros(FLAGS.max_epoch)
    g_loss_train = np.zeros(FLAGS.max_epoch)
    vae_loss_train = np.zeros(FLAGS.max_epoch)
    pub_dist_train = np.zeros(FLAGS.max_epoch)
    sec_dist_train = np.zeros(FLAGS.max_epoch)
    loss2x_train = np.zeros(FLAGS.max_epoch)
    loss2c_train = np.zeros(FLAGS.max_epoch)
    KLloss_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    g_loss_val = np.zeros(FLAGS.max_epoch)
    vae_loss_val = np.zeros(FLAGS.max_epoch)
    pub_dist_val = np.zeros(FLAGS.max_epoch)
    sec_dist_val = np.zeros(FLAGS.max_epoch)
    loss2x_val = np.zeros(FLAGS.max_epoch)
    loss2c_val = np.zeros(FLAGS.max_epoch)
    KLloss_val = np.zeros(FLAGS.max_epoch)
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
        KLv = 0
        loss2xv = 0
        loss2cv = 0
        loss3v = 0
        #pdb.set_trace()
        
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, True)
            vaelossv, zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([loss_vae, z, xhat, chat, mean, stddev, correct_pred], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            _, g_loss_value = sess.run([g_train, loss2], feeds)
            KLtmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([KLloss, loss2x, loss2c, loss_vae], feeds)
            if (np.isnan(e_loss_value) or np.isnan(g_loss_value)):
                pdb.set_trace()
                break
            if epoch == FLAGS.max_epoch-1:
                pdb.set_trace()
            #train_writer.add_summary(summary, i)
            e_training_loss += e_loss_value
            g_training_loss += g_loss_value
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            sec_accv += sec_acc_tmp
            KLv += KLtmp
            loss2xv += loss2xtmp
            loss2cv += loss2ctmp
            loss3v += loss2ctmp

        e_training_loss = e_training_loss / \
            (FLAGS.updates_per_epoch)
        g_training_loss = g_training_loss / \
            (FLAGS.updates_per_epoch)
        pub_loss /= (FLAGS.updates_per_epoch)
        sec_loss /= (FLAGS.updates_per_epoch)
        sec_accv /= (FLAGS.updates_per_epoch)
        loss2xv /= (FLAGS.updates_per_epoch)
        loss2cv /= (FLAGS.updates_per_epoch)
        loss3v /= (FLAGS.updates_per_epoch)
        KLv /= (FLAGS.updates_per_epoch)

        print("Loss for E %f, for G %f" % (e_training_loss, g_training_loss))
        print('Training public loss at epoch %s: %s' % (epoch, pub_loss))
        print('Training private loss at epoch %s: %s, private accuracy: %s' % (epoch, sec_loss, sec_accv))
        e_loss_train[epoch] = e_training_loss
        g_loss_train[epoch] = g_training_loss
        pub_dist_train[epoch] = pub_loss
        sec_dist_train[epoch] = sec_loss
        loss2x_train[epoch] = loss2xv
        loss2c_train[epoch] = loss2cv
        vae_loss_train[epoch] = loss3v
        KLloss_train[epoch] = KLv
        sec_acc_train[epoch] = sec_accv
        # Validation
        if epoch % 10 == 9:
                pub_loss = 0
                sec_loss = 0
                e_val_loss = 0
                g_val_loss = 0
                loss2xv = 0
                loss2cv = 0
                loss3v = 0
                KLv = 0
                sec_accv = 0

                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    feeds = get_feed(i, False)
                    pub_loss += sess.run(pub_dist, feeds)
                    sec_loss += sess.run(sec_dist, feeds)
                    e_val_loss += sess.run(loss1, feeds)
                    g_val_loss += sess.run(loss2, feeds)
                    KLtmp, loss2xtmp, loss2ctmp, sec_acc_tmp, loss3tmp = sess.run([KLloss, loss2x, loss2c, sec_acc, loss_vae], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        xhat_val.extend(sess.run(xhat, feeds))
                    #test_writer.add_summary(summary, i)
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    loss2xv += loss2xtmp
                    loss2cv += loss2ctmp
                    loss3v += loss3tmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                g_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss3v /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)

                print('Test public loss at epoch %s: %s' % (epoch, pub_loss))
                print('Test private loss at epoch %s: %s' % (epoch, sec_loss))
                e_loss_val[epoch] = e_val_loss
                g_loss_val[epoch] = g_val_loss
                pub_dist_val[epoch] = pub_loss
                sec_dist_val[epoch] = sec_loss
                loss2x_val[epoch] = loss2xv
                loss2c_val[epoch] = loss2cv
                vae_loss_val[epoch] = loss3v
                KLloss_val[epoch] = KLv
                sec_acc_val[epoch] = sec_accv
 
                if not(np.isnan(e_loss_value) or np.isnan(g_loss_value)):
                    savepath = saver.save(sess, model_directory + '/synth_privacy', global_step=epoch)
                    print('Model saved at epoch %s, path is %s' % (epoch, savepath))

    np.savez(os.path.join(model_directory, 'synth_trainstats'), e_loss_train=e_loss_train,
                                                  d_loss_train=g_loss_train,
                                                  pub_dist_train=pub_dist_train,
                                                  sec_dist_train=sec_dist_train,
                                                  loss2x_train = loss2x_train,
                                                  loss2c_train = loss2c_train,
                                                  vae_loss_train = vae_loss_train,
                                                  KLloss_train = KLloss_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  d_loss_val=g_loss_val,
                                                  pub_dist_val=pub_dist_val,
                                                  sec_dist_val=sec_dist_val,
                                                  loss2x_val = loss2x_val,
                                                  loss2c_val = loss2c_val,
                                                  vae_loss_val = vae_loss_val,
                                                  KLloss_val = KLloss_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

   

    sess.close()
    return

def train_gauss_discrim(prior):
    '''Train model to output transformation that prevents leaking private info, with weighted vector input data
    input: prior [1xM] probabilities of each class label in the dataset
    '''
    # Set a secret random seed for all models
    tf.set_random_seed(515319)


    FLAGS.dataset_size = 10000
    FLAGS.test_dataset_size = 5000
    FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
    FLAGS.input_size=10
    FLAGS.z_size=10
    FLAGS.output_size=10
    FLAGS.private_size=10
    FLAGS.hidden_size=20

    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic_weighted")
    model_directory = os.path.join(synth_dir, "discrim_privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])

    #load data
    data = np.load(synth_dir + '/weightedgaussian.npz')
    xs = data['x']
    cs = data['c']
    #convert class labels to one hot encoding
    cs = np.eye(np.max(cs)+1)[cs]
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        c = cs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        if x.shape==(0, 10):
            pdb.set_trace()
        return {input_tensor: x, output_tensor: x, private_tensor: c}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvib.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder") as scope:
                xhat, chat, mean, stddev = dvib.synth_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
            with tf.variable_scope("discrim") as scope:
                D1 = dvib.synth_discriminator(input_tensor) # positive samples
            with tf.variable_scope("discrim", reuse=True) as scope:
                D2 = dvib.synth_discriminator(xhat) # negative samples
                all_params = tf.trainable_variables()
                discrim_len = len(all_params) - (d_param_len + e_param_len)
    
    #Calculate losses
    _, KLloss = encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor)
    loss2x, loss2c = recon_cost(xhat, chat, output_tensor, private_tensor)
    loss_g = get_gen_cost(D2)
    loss_d = get_discrim_cost(D1, D2)
    loss1 = loss_g * encode_coef + KLloss
    loss2 = loss_g * decode_coef + loss2c
    loss_vae = get_vae_cost(mean, stddev)
    #loss1 = loss1 + encode_coef * loss3
    
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
        loss2xv = 0
        loss2cv = 0
        loss3v = 0
        #pdb.set_trace()
            
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, True)
            zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            for j in range(FLAGS.updates_per_epoch):
                g_feeds = get_feed(j, True)
                _, e_loss_value = sess.run([e_train, loss1], g_feeds)
                _, g_loss_value = sess.run([g_train, loss2], g_feeds)
            _, d_loss_value = sess.run([d_train, loss_d], feeds)
            KLtmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([KLloss, loss2x, loss2c, loss_vae], feeds)
            if (np.isnan(e_loss_value) or np.isnan(g_loss_value) or np.isnan(d_loss_value)):
                pdb.set_trace()
                break
            if epoch==FLAGS.max_epoch-1:
                pdb.set_trace()
            #train_writer.add_summary(summary, i)
            e_training_loss += e_loss_value
            g_training_loss += g_loss_value
            d_training_loss += d_loss_value
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            sec_accv += sec_acc_tmp
            KLv += KLtmp
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
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

   

    sess.close()
    return

def train_twotask_gauss(prior, lossmetric="KL", order=50):
    '''Train model to output transformation that prevents leaking private info, with vector input data
    input: prior [1xM] probabilities of each class label in the dataset
    '''
    # Set a secret random seed for all models
    tf.set_random_seed(515319)

    FLAGS.dataset_size = 10000
    FLAGS.test_dataset_size = 5000
    FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
    FLAGS.input_size=1
    FLAGS.z_size=4
    FLAGS.output_size=2
    FLAGS.private_size=2
    FLAGS.hidden_size=10

    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic")
    model_directory = os.path.join(synth_dir, "1D_2_2_twotask_privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef) + "_"+str(order))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])

    #load data
    data = np.load(synth_dir + '/1Dtwotaskgaussian_2_2.npz')
    xs = data['x']
    rawcs = data['c']
    ys = data['y']
    #convert class labels to one hot encoding
    cs = np.eye(np.max(rawcs)+1)[rawcs]
    ys = np.eye(np.max(ys)+1)[ys]
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        rawc = rawcs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        c = cs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        y = ys[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        return {input_tensor: x, output_tensor: y, private_tensor: c, rawc_tensor: rawc}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder", reuse=False):
                z = dvib.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder", reuse=False):
                yhat, chat, mean, stddev = dvib.synth_twotask_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
    
    #Calculate losses
    _, KLloss = dvibloss.encoding_cost(yhat, chat, output_tensor, private_tensor, prior_tensor, xmetric="CE", independent=False)
    loss2x, loss2c = dvibloss.recon_cost(yhat, chat, output_tensor, private_tensor, softmax=True, xmetric="CE")
    # Record losses of MI approximation and sibson MI
    h_c, h_cz, _ = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, yhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    # use alpha = 3 first, may be tuned
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, order, independent=False)
    # Compose losses
    if lossmetric=="KL":
        loss1 = encode_coef * loss2x + KLloss
    if lossmetric=="MI":
        loss1 = encode_coef * loss2x + I_c_cz
    if lossmetric=="sibMI":
        loss1 = encode_coef * loss2x + sibMI_c_cz
    loss2 = decode_coef * loss2x + loss2c
    loss3 = dvibloss.get_vae_cost(mean, stddev)
   
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean((yhat - output_tensor)**2)
            correct_predpub = tf.equal(tf.argmax(yhat, axis=1), tf.argmax(output_tensor, axis=1))
            pub_acc = tf.reduce_mean(tf.cast(correct_predpub, tf.float32))
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            pchat = tf.sigmoid(chat)
            sec_dist = tf.reduce_mean((pchat - private_tensor)**2)
            tmpchat = tf.concat([pchat, 1.0 - pchat], axis=1)
            tmppriv = tf.concat([private_tensor, 1.0 - private_tensor], axis=1)
            correct_pred = tf.equal(tf.argmax(tmpchat, axis=1), tf.argmax(tmppriv, axis=1))
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    e_train = pt.apply_optimizer(optimizer, losses=[loss1], regularize=True, include_marked=True, var_list=encode_params) # encoder training op
    d_train = pt.apply_optimizer(optimizer, losses=[loss2], regularize=True, include_marked=True, var_list=all_params[e_param_len:]) # generator/decoder training op
    # Logging matrices
    e_loss_train = np.zeros(FLAGS.max_epoch)
    d_loss_train = np.zeros(FLAGS.max_epoch)
    vae_loss_train = np.zeros(FLAGS.max_epoch)
    pub_dist_train = np.zeros(FLAGS.max_epoch)
    sec_dist_train = np.zeros(FLAGS.max_epoch)
    loss2x_train = np.zeros(FLAGS.max_epoch)
    loss2c_train = np.zeros(FLAGS.max_epoch)
    KLloss_train = np.zeros(FLAGS.max_epoch)
    MIloss_train = np.zeros(FLAGS.max_epoch)
    sibMIloss_train = np.zeros(FLAGS.max_epoch)
    pub_acc_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    d_loss_val = np.zeros(FLAGS.max_epoch)
    vae_loss_val = np.zeros(FLAGS.max_epoch)
    pub_dist_val = np.zeros(FLAGS.max_epoch)
    sec_dist_val = np.zeros(FLAGS.max_epoch)
    loss2x_val = np.zeros(FLAGS.max_epoch)
    loss2c_val = np.zeros(FLAGS.max_epoch)
    KLloss_val = np.zeros(FLAGS.max_epoch)
    MIloss_val = np.zeros(FLAGS.max_epoch)
    sibMIloss_val = np.zeros(FLAGS.max_epoch)
    pub_acc_val = np.zeros(FLAGS.max_epoch)
    sec_acc_val = np.zeros(FLAGS.max_epoch)
    yhat_val = []
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
        pub_accv = 0
        sec_accv = 0
        e_training_loss = 0
        d_training_loss = 0
        KLv = 0
        MIv = 0
        sibMIv = 0
        loss2xv = 0
        loss2cv = 0
        loss3v = 0
        #pdb.set_trace()
        
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, True)
            #vaelossv, zv, yhatv, chatv, meanv, stddevv, sec_pred = sess.run([loss3, z, yhat, chat, mean, stddev, correct_pred], feeds)
            pub_tmp, sec_tmp, pub_acc_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, pub_acc, sec_acc], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            _, d_loss_value = sess.run([d_train, loss2], feeds)
            KLtmp, MItmp, sibMItmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([KLloss, I_c_cz, sibMI_c_cz, loss2x, loss2c, loss3], feeds)
            if (np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                pdb.set_trace()
                break
            #if epoch == FLAGS.max_epoch-1:
            #    pdb.set_trace()
            #train_writer.add_summary(summary, i)
            e_training_loss += e_loss_value
            d_training_loss += d_loss_value
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            pub_accv += pub_acc_tmp
            sec_accv += sec_acc_tmp
            KLv += KLtmp
            MIv += MItmp
            sibMIv += sibMItmp
            loss2xv += loss2xtmp
            loss2cv += loss2ctmp
            loss3v += loss2ctmp

        e_training_loss = e_training_loss / \
            (FLAGS.updates_per_epoch)
        d_training_loss = d_training_loss / \
            (FLAGS.updates_per_epoch)
        pub_loss /= (FLAGS.updates_per_epoch)
        sec_loss /= (FLAGS.updates_per_epoch)
        pub_accv /= (FLAGS.updates_per_epoch)
        sec_accv /= (FLAGS.updates_per_epoch)
        loss2xv /= (FLAGS.updates_per_epoch)
        loss2cv /= (FLAGS.updates_per_epoch)
        loss3v /= (FLAGS.updates_per_epoch)
        KLv /= (FLAGS.updates_per_epoch)
        MIv /= (FLAGS.updates_per_epoch)
        sibMIv /= (FLAGS.updates_per_epoch)

        print("Loss for E %f, for D %f" % (e_training_loss, d_training_loss))
        print('Training public loss at epoch %s: %s, public accuracy: %s' % (epoch, pub_loss, pub_accv))
        print('Training private loss at epoch %s: %s, private accuracy: %s' % (epoch, sec_loss, sec_accv))
        e_loss_train[epoch] = e_training_loss
        d_loss_train[epoch] = d_training_loss
        pub_dist_train[epoch] = pub_loss
        sec_dist_train[epoch] = sec_loss
        loss2x_train[epoch] = loss2xv
        loss2c_train[epoch] = loss2cv
        vae_loss_train[epoch] = loss3v
        KLloss_train[epoch] = KLv
        MIloss_train[epoch] = MIv
        sibMIloss_train[epoch] = sibMIv
        pub_acc_train[epoch] = pub_accv
        sec_acc_train[epoch] = sec_accv
        # Validation
        if epoch % 10 == 9:
                pub_loss = 0
                sec_loss = 0
                e_val_loss = 0
                d_val_loss = 0
                loss2xv = 0
                loss2cv = 0
                loss3v = 0
                KLv = 0
                MIv = 0
                sibMIv = 0
                pub_accv = 0
                sec_accv = 0

                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    feeds = get_feed(i, False)
                    pub_loss += sess.run(pub_dist, feeds)
                    sec_loss += sess.run(sec_dist, feeds)
                    e_val_loss += sess.run(loss1, feeds)
                    d_val_loss += sess.run(loss2, feeds)
                    KLtmp, MItmp, sibMItmp, loss2xtmp, loss2ctmp, sec_acc_tmp, loss3tmp = sess.run([KLloss, I_c_cz, sibMI_c_cz, loss2x, loss2c, sec_acc, loss3], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        yhat_val.extend(sess.run(yhat, feeds))
                    #test_writer.add_summary(summary, i)
                    pub_accv += pub_acc_tmp
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    MIv += MItmp
                    sibMIv += sibMItmp
                    loss2xv += loss2xtmp
                    loss2cv += loss2ctmp
                    loss3v += loss3tmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                d_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss3v /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                MIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sibMIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                pub_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)

                print('Test public loss at epoch %s: %s, public accuracy: %s' % (epoch, pub_loss, pub_accv))
                print('Test private loss at epoch %s: %s, private accuracy: %s' % (epoch, sec_loss, sec_accv))
                e_loss_val[epoch] = e_val_loss
                d_loss_val[epoch] = d_val_loss
                pub_dist_val[epoch] = pub_loss
                sec_dist_val[epoch] = sec_loss
                loss2x_val[epoch] = loss2xv
                loss2c_val[epoch] = loss2cv
                vae_loss_val[epoch] = loss3v
                KLloss_val[epoch] = KLv
                MIloss_val[epoch] = MIv
                sibMIloss_val[epoch] = sibMIv
                pub_acc_val[epoch] = pub_accv
                sec_acc_val[epoch] = sec_accv
 
                if not(np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                    savepath = saver.save(sess, model_directory + '/twotask_privacy', global_step=epoch)
                    print('Model saved at epoch %s, path is %s' % (epoch, savepath))

    np.savez(os.path.join(model_directory, 'synth_trainstats'), e_loss_train=e_loss_train,
                                                  d_loss_train=d_loss_train,
                                                  pub_dist_train=pub_dist_train,
                                                  sec_dist_train=sec_dist_train,
                                                  loss2x_train = loss2x_train,
                                                  loss2c_train = loss2c_train,
                                                  vae_loss_train = vae_loss_train,
                                                  KLloss_train = KLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sibMIloss_train = sibMIloss_train,
                                                  pub_acc_train = pub_acc_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  d_loss_val=d_loss_val,
                                                  pub_dist_val=pub_dist_val,
                                                  sec_dist_val=sec_dist_val,
                                                  loss2x_val = loss2x_val,
                                                  loss2c_val = loss2c_val,
                                                  vae_loss_val = vae_loss_val,
                                                  KLloss_val = KLloss_val,
                                                  MIloss_val = MIloss_val,
                                                  sibMIloss_val = sibMIloss_val,
                                                  pub_acc_val = pub_acc_val,
                                                  sec_acc_val = sec_acc_val,
                                                  yhat_val = yhat_val
)

   

    sess.close()
    return

def train_2gauss_CGAP(prior, lossmetric="KL", order=20, K_iters=50, D=0.1):
    '''Train model to output transformation that prevents leaking private info
    based on the CGAP algorithm as a baseline
    '''
    # Set a secret random seed for all models
    tf.set_random_seed(515319)

    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "synthetic")
    model_directory = os.path.join(synth_dir, lossmetric+"_"+"CGAP_privacy_checkpoints"+str(K_iters)+"_"+str(D)+"_"+str(order))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    rou_tensor = tf.placeholder(tf.float32)
    D_tensor = tf.placeholder(tf.float32)
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])
    # Initialize vectors for class dependent affine transformation
    #pdb.set_trace()
    beta0 = weight_variable([1, FLAGS.z_size], None)
    beta1 = weight_variable([1, FLAGS.z_size], None)
    gamma0 = weight_variable([1, FLAGS.z_size], None)
    gamma1 = weight_variable([1, FLAGS.z_size], None)

    #load data, if ends in _3_m3 then means are [3, -3]
    data = np.load(synth_dir + '/1d2gaussian_3_m3.npz')
    xs = data['x']
    cs = data['c']
    def get_feed(batch_no, training):
        offset = FLAGS.dataset_size if training==False else 0
        x = xs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        pow_x = np.array([x, x**2, x**3]).transpose()
        x = np.array(x).reshape(FLAGS.batch_size, 1)
        c = cs[offset + FLAGS.batch_size * batch_no: offset + FLAGS.batch_size * (batch_no + 1)]
        c = np.array(c).reshape(FLAGS.batch_size, 1)
        rawc = c.reshape(FLAGS.batch_size)
        return {input_tensor: x, output_tensor: x, private_tensor: c, rawc_tensor: rawc}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder", reuse=False) as scope:
                z = dvib.synth_encoder(input_tensor, private_tensor, FLAGS.hidden_size)
                #z = dvib.synth_affine_encoder(input_tensor, private_tensor, beta0, beta1)
                #z = dvib.synth_affine_noisy_encoder(input_tensor, private_tensor, beta0, beta1, gamma0, gamma1)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder", reuse=False) as scope:
                xhat, chat, mean, stddev = dvib.synth_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len

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
    loss3 = dvibloss.get_vae_cost(mean, stddev)
    _, KLloss = dvibloss.encoding_cost(xhat, chat, output_tensor, private_tensor, prior_tensor, xmetric="CE", independent=False)
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, order, independent=False)
    h_c, h_cz, _ = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    #Calculate the loss for CGAP, using the log loss as specified
    ptilde = 0.5
    lossdist = ptilde * (beta0**2 + gamma0**2) + (1 - ptilde) * (beta1**2 + gamma1**2)
    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=private_tensor, logits=chat)) + rou_tensor * tf.maximum(0.0, lossdist - D_tensor)
    loss2 = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=private_tensor, logits=chat)) + rou_tensor * tf.maximum(0.0, lossdist - D_tensor)
    #loss = sibMI_c_cz + rou_tensor * tf.maximum(0.0, lossdist - D_tensor)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    adv_train = optimizer.minimize(loss1, var_list=encode_params)
    priv_train = optimizer.minimize(loss2, var_list=all_params[e_param_len:])
 
    # Logging matrices
    e_loss_train = np.zeros(FLAGS.max_epoch)
    d_loss_train = np.zeros(FLAGS.max_epoch)
    pub_dist_train = np.zeros(FLAGS.max_epoch)
    sec_dist_train = np.zeros(FLAGS.max_epoch)
    loss2x_train = np.zeros(FLAGS.max_epoch)
    loss2c_train = np.zeros(FLAGS.max_epoch)
    KLloss_train = np.zeros(FLAGS.max_epoch)
    npKLloss_train = np.zeros(FLAGS.max_epoch)
    MIloss_train = np.zeros(FLAGS.max_epoch)
    sibMIloss_train = np.zeros(FLAGS.max_epoch)
    npsibMIloss_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    d_loss_val = np.zeros(FLAGS.max_epoch)
    pub_dist_val = np.zeros(FLAGS.max_epoch)
    sec_dist_val = np.zeros(FLAGS.max_epoch)
    loss2x_val = np.zeros(FLAGS.max_epoch)
    loss2c_val = np.zeros(FLAGS.max_epoch)
    KLloss_val = np.zeros(FLAGS.max_epoch)
    MIloss_val = np.zeros(FLAGS.max_epoch)
    sibMIloss_val = np.zeros(FLAGS.max_epoch)
    sec_acc_val = np.zeros(FLAGS.max_epoch)
    xhat_val = []
    # Tensorboard logging
    #tf.summary.scalar('e_loss', loss1)
    #tf.summary.scalar('KL', KLloss)
    #tf.summary.scalar('loss_x', loss2x)
    #tf.summary.scalar('loss_c', loss2c)
    #tf.summary.scalar('pub_dist', pub_dist)
    #tf.summary.scalar('sec_dist', sec_dist)
    # Rou tensor values, penalty parameter for the distortion constraint
    rou_values = np.linspace(0, 1000, FLAGS.max_epoch)



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
        npKLv = 0
        sibMIv = 0
        npsibMIv = 0
        loss2xv = 0
        loss2cv = 0
        #if epoch == FLAGS.max_epoch-1:
        #    pdb.set_trace()
            
        #pdb.set_trace()
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            # Feed values of data and rou, D parameters
            feeds = get_feed(i, True)
            feeds[rou_tensor] = rou_values[epoch]
            feeds[D_tensor] = D
            zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            d_loss_inner = 0
            for j in range(K_iters):
                _, d_loss_value = sess.run([priv_train, loss1], feeds)
                d_loss_inner += d_loss_value
            _, e_loss_value = sess.run([adv_train, loss2], feeds)
            MItmp, sibMItmp, KLtmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([I_c_cz, sibMI_c_cz, KLloss, loss2x, loss2c, loss3], feeds)
            pchatv = 1.0 / (1.0 + np.exp(-chatv))
            npKLtmp = np.average(pchatv * np.log(np.clip(pchatv/prior[0], eps, None))) + np.average((1.0-pchatv) * np.log(np.clip((1.0-pchatv)/(1.0-prior[0]), eps, None)))
            npsibMItmp = (order/(order-1.0)) * np.log(np.clip(np.average(pchatv**order)**(1.0/order) + np.average((1.0-pchatv)**order)**(1.0/order), eps, None))
 
            if (np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                pdb.set_trace()
                break
            #train_writer.add_summary(summary, i)
            e_training_loss += e_loss_value
            d_training_loss += (d_loss_inner/K_iters)
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            sec_accv += sec_acc_tmp
            KLv += KLtmp
            npKLv += npKLtmp
            MIv += MItmp
            sibMIv += npsibMItmp
            npsibMIv += sibMItmp
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
        npKLv /= (FLAGS.updates_per_epoch)
        MIv /= (FLAGS.updates_per_epoch)
        sibMIv /= (FLAGS.updates_per_epoch)
        npsibMIv /= (FLAGS.updates_per_epoch)

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
        npKLloss_train[epoch] = npKLv
        sibMIloss_train[epoch] = sibMIv
        npsibMIloss_train[epoch] = npsibMIv
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
                sibMIv = 0
                sec_accv = 0

                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    feeds = get_feed(i, False)
                    feeds[rou_tensor] = rou_values[epoch]
                    feeds[D_tensor] = D
                    pub_loss += sess.run(pub_dist, feeds)
                    sec_loss += sess.run(sec_dist, feeds)
                    e_val_loss += sess.run(loss2, feeds)
                    d_val_loss += sess.run(loss1, feeds)
                    MItmp, sibMItmp, KLtmp, loss2xtmp, loss2ctmp, sec_acc_tmp = sess.run([I_c_cz, sibMI_c_cz, KLloss, loss2x, loss2c, sec_acc], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        xhat_val.extend(sess.run(xhat, feeds))
                    #test_writer.add_summary(summary, i)
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    MIv += MItmp
                    sibMIv += sibMItmp
                    loss2xv += loss2xtmp
                    loss2cv += loss2ctmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                d_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                MIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sibMIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_accv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)

                print('Test public loss at epoch %s: %s' % (epoch, pub_loss))
                print('Test private loss at epoch %s: %s' % (epoch, sec_loss))
                print('Test CGAP loss at epoch %s: %s' % (epoch, e_val_loss))
                e_loss_val[epoch] = e_val_loss
                d_loss_val[epoch] = d_val_loss
                pub_dist_val[epoch] = pub_loss
                sec_dist_val[epoch] = sec_loss
                loss2x_val[epoch] = loss2xv
                loss2c_val[epoch] = loss2cv
                KLloss_val[epoch] = KLv
                MIloss_val[epoch] = MIv
                sibMIloss_val[epoch] = sibMIv
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
                                                  npKLloss_train = npKLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sibMIloss_train = sibMIloss_train,
                                                  npsibMIloss_train = npsibMIloss_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  d_loss_val=d_loss_val,
                                                  pub_dist_val=pub_dist_val,
                                                  sec_dist_val=sec_dist_val,
                                                  loss2x_val = loss2x_val,
                                                  loss2c_val = loss2c_val,
                                                  KLloss_val = KLloss_val,
                                                  MIloss_val = MIloss_val,
                                                  sibMIloss_val = sibMIloss_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

    sess.close()
    return

def train_twotask_model(prior):
    pdb.set_trace()
    range_encode = np.concatenate([np.linspace(0.1, 1, 11), [2, 5, 10, 20]])
    #range_encode = np.linspace(2,15, 14)
    for encode_iter_i in range_encode:
        encode_coef = encode_iter_i
        train_twotask_gauss(prior, lossmetric="sibMI")
    return

if __name__ == '__main__':
    #pdb.set_trace()
    #prior = calc_pc()
    #encode_coef = 1e-3
    #pc = calc_pc_weighted()
    #train_gauss_weighted(pc)
    #train_gauss_discrim(pc)
    # Two task synthetic data experiment
    #prior = calc_pc(data=np.load(FLAGS.working_directory+"/data/synthetic/1Dtwotaskgaussian_2_2.npz"))
    #train_twotask_gauss(prior, lossmetric="sibMI")
    
    # CGAP experiment
    prior = calc_pc(data=np.load(FLAGS.working_directory+"/data/synthetic/1d2gaussian_3_m3.npz"))
    train_2gauss_CGAP(prior, lossmetric="sibMI")
    # Comparison with theoretic results
    train_2gauss_compare_theoretic(prior)
    # Older experiment with encode/decode coefficients
    #train_2gauss(prior, lossmetric="sibMI")
    #train_twotask_model(prior)

