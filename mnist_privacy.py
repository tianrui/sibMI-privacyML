from __future__ import absolute_import, division, print_function

import math
import os
import gc

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
import pdb
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
#working directory for home
#FLAGS.working_directory = "/home/rxiao"
#working directory for UG computer
FLAGS.working_directory = "/cad2/ece521s/cuda_libs/mnist"
FLAGS.summary_dir = os.path.join(os.getcwd(), 'synth')
FLAGS.dataset_size = 50000
FLAGS.test_dataset_size = 10000
FLAGS.batch_size = 500
FLAGS.updates_per_epoch = int(FLAGS.dataset_size / FLAGS.batch_size)
FLAGS.max_epoch = 200
FLAGS.keep_prob = 0.8
FLAGS.learning_rate = 2e-3
FLAGS.hidden_size = 512
FLAGS.input_size = 784
FLAGS.z_size = 128
FLAGS.output_size = 784
FLAGS.private_size = 10
FLAGS.restore_model = False

eps = 1e-16
# loss1 = loss1x * encode_coef + KL
# loss2 = loss2x * decode_coef + binCE
encode_coef = 1
decode_coef = 1

def calc_pc():
    '''Calculate prior probabiltiy of the private variable
    '''
    #pdb.set_trace()
    data_dir = os.path.join(FLAGS.working_directory, "data")
    synth_dir = os.path.join(data_dir, "mnist")
    mnist = input_data.read_data_sets(synth_dir, one_hot=True)
    cs = mnist.train.labels
    counts = np.sum(cs, axis=0)/cs.shape[0]
    return counts

def train_mnist(prior, lossmetric="sibMI", order=20, D=0.02, xmetric="L2", K_iters=20):
    '''Train model to output transformation that prevents leaking private info
    '''
    # Set random seed for this model
    tf.set_random_seed(515319)

    data_dir = os.path.join(FLAGS.working_directory, "data")
    mnist_dir = os.path.join(data_dir, "mnist")
    model_directory = os.path.join(mnist_dir, lossmetric+"privacy_checkpoints"+str(encode_coef)+"_"+str(decode_coef)+"_D"+str(D)+"_order"+str(order)+"_xmetric"+xmetric)
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])
    rou_tensor = tf.placeholder(tf.float32)
    D_tensor = tf.placeholder(tf.float32)

    #load data not necessary for mnist data, formatted as vectors of real values between 0 and 1
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    def get_feed(batch_no, training):
        if training:
            x, c = mnist.train.next_batch(FLAGS.batch_size)
        else:
            x, c = mnist.test.next_batch(FLAGS.batch_size)
        rawc = np.argmax(c, axis=1)
        return {input_tensor: x, output_tensor: x, private_tensor: c[:, :FLAGS.private_size], rawc_tensor: rawc}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvibcomp.privacy_encoder(input_tensor, private_tensor)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder") as scope:
                xhat, chat, mean, stddev = dvibcomp.mnist_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
   
    # Calculating losses 
    _, KLloss = dvibloss.encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor, xmetric=xmetric, independent=False)
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, input_tensor, private_tensor, softmax=True, xmetric=xmetric)
    # Record losses of MI approximation and sibson MI
    h_c, h_cz, _ = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    # Ialpha(Z;C)
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, order, independent=False)
    # Ialpha(C;Z)
    sibMI_c_z = dvibloss.sibsonMI_c_z(z, chat, prior_tensor, order, independent=False)
    # Distortion constraint
    lossdist = rou_tensor * tf.maximum(0.0, loss2x - D_tensor)
    # Compose losses
    if lossmetric=="KL":
        loss1 = encode_coef * lossdist + KLloss
    if lossmetric=="MI":
        loss1 = encode_coef * lossdist + I_c_cz
    if lossmetric=="sibMI":
        loss1 = encode_coef * lossdist + sibMI_c_z
    loss2 = decode_coef * lossdist + loss2c
    loss3 = dvibloss.get_vae_cost(mean, stddev)
    
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean((xhat - output_tensor)**2)
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            sec_dist = tf.reduce_mean((chat - private_tensor)**2)
            #correct_pred = tf.less(tf.abs(chat - private_tensor), 0.5)
            correct_pred = tf.equal(tf.argmax(chat, axis=1), tf.argmax(private_tensor, axis=1))
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
    sibMIloss_train = np.zeros(FLAGS.max_epoch)
    sibMIcz_train = np.zeros(FLAGS.max_epoch)
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
    sibMIcz_val = np.zeros(FLAGS.max_epoch)
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
    rou_values = np.linspace(1, 1000, FLAGS.max_epoch)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False

    sess = tf.Session(config=config)
    sess.run(init)
    #merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train', sess.graph)
    #test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
    
    #Attempt to restart from last checkpt
    checkpt = tf.train.latest_checkpoint(model_directory)
    if checkpt != None and FLAGS.restore_model==True:
        saver.restore(sess, checkpt)
        print("Restored model from checkpoint %s" % (checkpt))
 
 
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
        sibMIv = 0
        sibMIczv = 0
        loss2xv = 0
        loss2cv = 0
        #pdb.set_trace()
        #if epoch == FLAGS.max_epoch-1:
            #pdb.set_trace()
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, True)
            feeds[rou_tensor] = rou_values[epoch]
            feeds[D_tensor] = D
            #zv, xhatv, chatv, meanv, stddevv, sec_pred = sess.run([z, xhat, chat, mean, stddev, correct_pred], feeds)
            pub_tmp, sec_tmp, sec_acc_tmp = sess.run([pub_dist, sec_dist, sec_acc], feeds)
            MItmp, sibMItmp, sibMIcztmp, KLtmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([I_c_cz, sibMI_c_cz, sibMI_c_z, KLloss, loss2x, loss2c, loss3], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            d_inner = 0
            for j in range(K_iters):
                _, d_inner = sess.run([d_train, loss2], feeds)
            d_loss_value = d_inner/K_iters
            if (np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                #pdb.set_trace()
                break
            #train_writer.add_summary(summary, i)
            e_training_loss += e_loss_value
            d_training_loss += d_loss_value
            pub_loss += pub_tmp
            sec_loss += sec_tmp
            sec_accv += sec_acc_tmp
            KLv += KLtmp
            MIv += MItmp
            sibMIv += sibMItmp
            sibMIczv += sibMIcztmp
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
        sibMIv /= (FLAGS.updates_per_epoch)
        sibMIczv /= (FLAGS.updates_per_epoch)

        print("Loss for E %f, and for D %f" % (e_training_loss, d_training_loss))
        print('Training public loss at epoch %s: %s' % (epoch, pub_loss))
        print('Training private loss at epoch %s: %s, private accuracy: %s' % (epoch, sec_loss, sec_accv))
        print('Training KL loss at epoch %s: %s, sibMI(Z;C): %s, sibMI(C;Z): %s, loss2x: %s' % (epoch, KLv, sibMIv, sibMIczv, loss2xv))
        if sibMIv < 0:
            print("sibson MI calculation breakdown: %s" % (sibMIv)) 
            savepath = saver.save(sess, model_directory + '/mnist_privacy', global_step=epoch)
            print('Model saved at epoch %s, path is %s' % (epoch, savepath))

            np.savez(os.path.join(model_directory, 'synth_trainstats'), e_loss_train=e_loss_train,
                                                  d_loss_train=d_loss_train,
                                                  pub_dist_train=pub_dist_train,
                                                  sec_dist_train=sec_dist_train,
                                                  loss2x_train = loss2x_train,
                                                  loss2c_train = loss2c_train,
                                                  KLloss_train = KLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sibMIloss_train = sibMIloss_train,
                                                  sibMIcz_train = sibMIcz_train,
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
                                                  sibMIcz_val = sibMIcz_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)
            break
        e_loss_train[epoch] = e_training_loss
        d_loss_train[epoch] = d_training_loss
        pub_dist_train[epoch] = pub_loss
        sec_dist_train[epoch] = sec_loss
        loss2x_train[epoch] = loss2xv
        loss2c_train[epoch] = loss2cv
        KLloss_train[epoch] = KLv
        MIloss_train[epoch] = MIv
        sibMIloss_train[epoch] = sibMIv
        sibMIcz_train[epoch] = sibMIczv
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
                sibMIczv = 0
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
                    MItmp, sibMItmp, sibMIcztmp, KLtmp, loss2xtmp, loss2ctmp, sec_acc_tmp = sess.run([I_c_cz, sibMI_c_cz, sibMI_c_z, KLloss, loss2x, loss2c, sec_acc], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        xhat_val.extend(sess.run(xhat, feeds))
                    #test_writer.add_summary(summary, i)
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    MIv += MItmp
                    sibMIv += sibMItmp
                    sibMIczv += sibMIcztmp
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
                sibMIczv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
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
                sibMIloss_val[epoch] = sibMIv
                sibMIcz_val[epoch] = sibMIczv
                sec_acc_val[epoch] = sec_accv
 
                if not(np.isnan(e_loss_value) or np.isnan(d_loss_value)):
                    savepath = saver.save(sess, model_directory + '/mnist_privacy', global_step=epoch)
                    print('Model saved at epoch %s, path is %s' % (epoch, savepath))

    np.savez(os.path.join(model_directory, 'synth_trainstats'), e_loss_train=e_loss_train,
                                                  d_loss_train=d_loss_train,
                                                  pub_dist_train=pub_dist_train,
                                                  sec_dist_train=sec_dist_train,
                                                  loss2x_train = loss2x_train,
                                                  loss2c_train = loss2c_train,
                                                  KLloss_train = KLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sibMIloss_train = sibMIloss_train,
                                                  sibMIcz_train = sibMIcz_train,
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
                                                  sibMIcz_val = sibMIcz_val,
                                                  sec_acc_val = sec_acc_val,
                                                  xhat_val = xhat_val
)

    sess.close()

def train_mnist_discrim(prior, lossmetric="KL"):
    '''Train model to output transformation that prevents leaking private info
       using a discriminator to aid producing natural images
    '''
    # Set random seed for this model
    tf.random.set_random_seed(515319)

    data_dir = os.path.join(FLAGS.working_directory, "data")
    mnist_dir = os.path.join(data_dir, "mnist")
    model_directory = os.path.join(mnist_dir, lossmetric+"discrim_privacy_checkpoints"+str(encode_coef))
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])

    #load data not necessary for mnist data
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    def get_feed(batch_no, training):
        if training:
            x, c = mnist.train.next_batch(FLAGS.batch_size)
        else:
            x, c = mnist.test.next_batch(FLAGS.batch_size)
        rawc = np.argmax(c, axis=1)
        return {input_tensor: x, output_tensor: x, private_tensor: c[:, :FLAGS.private_size], rawc_tensor: rawc}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder") as scope:
                z = dvibcomp.privacy_encoder(input_tensor, private_tensor)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder") as scope:
                xhat, chat, mean, stddev = dvibcomp.mnist_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
            with tf.variable_scope("discrim") as scope:
                D1 = dvibcomp.mnist_discriminator(input_tensor) # positive samples
            with tf.variable_scope("discrim", reuse=True) as scope:
                D2 = dvibcomp.mnist_discriminator(xhat) # negative samples
                all_params = tf.trainable_variables()
                discrim_len = len(all_params) - (d_param_len + e_param_len)
    
    # Calculating losses 
    _, KLloss = dvibloss.encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor)
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, input_tensor, private_tensor, softmax=True)
    loss_g = dvibloss.get_gen_cost(D2)
    loss_d = dvibloss.get_discrim_cost(D1, D2)
    loss_vae = dvibloss.get_vae_cost(mean, stddev)
    # Record losses of MI approximation and sibson MI
    h_c, h_cz, _, _ = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    I_c_cz = tf.abs(h_c - h_cz)
    # use alpha = 3 first, may be tuned
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, 3)
    # Compose losses
    if lossmetric=="KL":
        loss1 = encode_coef * loss_g + KLloss
    if lossmetric=="MI":
        loss1 = encode_coef * loss_g + I_c_cz
    if lossmetric=="sibMI":
        loss1 = encode_coef * loss_g + sibMI_c_cz
    loss2 = decode_coef * loss_g + loss2c
    loss3 = loss_d
   
    with tf.name_scope('pub_prediction'):
        with tf.name_scope('pub_distance'):
            pub_dist = tf.reduce_mean((xhat - output_tensor)**2)
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            sec_dist = tf.reduce_mean((chat - private_tensor)**2)
            #correct_pred = tf.less(tf.abs(chat - private_tensor), 0.5)
            correct_pred = tf.equal(tf.argmax(chat, axis=1), tf.argmax(private_tensor, axis=1))
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    e_train = pt.apply_optimizer(optimizer, losses=[loss1], regularize=True, include_marked=True, var_list=encode_params) # privatizer/encoder training op
    g_train = pt.apply_optimizer(optimizer, losses=[loss2], regularize=True, include_marked=True, var_list=all_params[e_param_len:]) # generator/decoder training op
    d_train = pt.apply_optimizer(optimizer, losses=[loss3], regularize=True, include_marked=True, var_list=all_params[e_param_len+d_param_len:]) # discriminator training op
    # Logging matrices
    e_loss_train = np.zeros(FLAGS.max_epoch)
    g_loss_train = np.zeros(FLAGS.max_epoch)
    d_loss_train = np.zeros(FLAGS.max_epoch)
    pub_dist_train = np.zeros(FLAGS.max_epoch)
    sec_dist_train = np.zeros(FLAGS.max_epoch)
    loss2x_train = np.zeros(FLAGS.max_epoch)
    loss2c_train = np.zeros(FLAGS.max_epoch)
    KLloss_train = np.zeros(FLAGS.max_epoch)
    MIloss_train = np.zeros(FLAGS.max_epoch)
    sibMIloss_train = np.zeros(FLAGS.max_epoch)
    sec_acc_train = np.zeros(FLAGS.max_epoch)
    e_loss_val = np.zeros(FLAGS.max_epoch)
    g_loss_val = np.zeros(FLAGS.max_epoch)
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
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False

    sess = tf.Session(config=config)
    sess.run(init)
 
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
        sibMIv = 0
        loss2xv = 0
        loss2cv = 0
            
        for i in range(FLAGS.updates_per_epoch):
            pbar.update(i)
            feeds = get_feed(i, True)
            pub_tmp, sec_tmp, sec_acc_tmp, KLtmp, MItmp, sibMItmp, loss2xtmp, loss2ctmp, loss3tmp = sess.run([pub_dist, sec_dist, sec_acc, KLloss, I_c_cz, sibMI_c_cz, loss2x, loss2c, loss_vae], feeds)
            _, e_loss_value = sess.run([e_train, loss1], feeds)
            _, g_loss_value = sess.run([g_train, loss2], feeds)
            _, d_loss_value = sess.run([d_train, loss3], feeds)
            if (np.isnan(e_loss_value) or np.isnan(g_loss_value)or np.isnan(d_loss_value)):
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
            MIv += MItmp
            sibMIv += sibMItmp
            loss2xv += loss2xtmp
            loss2cv += loss2ctmp

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
        KLv /= (FLAGS.updates_per_epoch)
        MIv /= (FLAGS.updates_per_epoch)
        sibMIv /= (FLAGS.updates_per_epoch)

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
        KLloss_train[epoch] = KLv
        MIloss_train[epoch] = MIv
        sibMIloss_train[epoch] = sibMIv
        sec_acc_train[epoch] = sec_accv
        # Forced Garbage Collection
        gc.collect()
        # Validation
        if epoch % 10 == 9:
                pub_loss = 0
                sec_loss = 0
                e_val_loss = 0
                g_val_loss = 0
                d_val_loss = 0
                loss2xv = 0
                loss2cv = 0
                KLv = 0
                MIv = 0
                sec_accv = 0

                for i in range(int(FLAGS.test_dataset_size / FLAGS.batch_size)):
                    feeds = get_feed(i, False)
                    e_val_tmp, g_val_tmp, d_val_tmp, pub_loss, sec_loss, MItmp, sibMItmp, KLtmp, loss2xtmp, loss2ctmp, sec_acc_tmp = sess.run([loss1, loss2, loss3, pub_dist, sec_dist, I_c_cz, sibMI_c_cz, KLloss, loss2x, loss2c, sec_acc], feeds)
                    if (epoch >= FLAGS.max_epoch - 10):
                        xhat_val.extend(sess.run(xhat, feeds))
                    #test_writer.add_summary(summary, i)
                    e_val_loss += e_val_tmp
                    g_val_loss += g_val_tmp
                    d_val_loss += d_val_tmp
                    sec_accv += sec_acc_tmp
                    KLv += KLtmp
                    MIv += MItmp
                    sibMIv += sibMItmp
                    loss2xv += loss2xtmp
                    loss2cv += loss2ctmp

                pub_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sec_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                e_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                g_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                d_val_loss /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2xv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                loss2cv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                KLv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                MIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
                sibMIv /= int(FLAGS.test_dataset_size / FLAGS.batch_size)
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
                KLloss_val[epoch] = KLv
                MIloss_val[epoch] = MIv
                sibMIloss_val[epoch] = sibMIv
                sec_acc_val[epoch] = sec_accv
 
                if not(np.isnan(e_val_loss) or np.isnan(g_val_loss)or np.isnan(d_val_loss)):
                    savepath = saver.save(sess, model_directory + '/mnist_privacy', global_step=epoch)
                    print('Model saved at epoch %s, path is %s' % (epoch, savepath))
                    gc.collect()

    np.savez(os.path.join(model_directory, 'synth_trainstats'), e_loss_train=e_loss_train,
                                                  g_loss_train=g_loss_train,
                                                  d_loss_train=d_loss_train,
                                                  pub_dist_train=pub_dist_train,
                                                  sec_dist_train=sec_dist_train,
                                                  loss2x_train = loss2x_train,
                                                  loss2c_train = loss2c_train,
                                                  KLloss_train = KLloss_train,
                                                  MIloss_train = MIloss_train,
                                                  sibMIloss_train = sibMIloss_train,
                                                  sec_acc_train = sec_acc_train,
                                                  e_loss_val=e_loss_val,
                                                  g_loss_val=g_loss_val,
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

def eval_checkpt(encode_coef, lossmetric="KL", D="0.12"):
    # Set random seed for this model
    tf.random.set_random_seed(515319)


    prior = calc_pc()
    pdb.set_trace()
    data_dir = os.path.join(FLAGS.working_directory, "data")
    mnist_dir = os.path.join(data_dir, "mnist")
    #model_directory = os.path.join(mnist_dir, lossmetric+"privacy_checkpoints"+str(encode_coef))
    #if lossmetric == "sibMI":
    #model_directory = os.path.join(mnist_dir, "remote_exp/"+lossmetric+"privacy_checkpoints1_1_D"+D+"_order20_xmetricPPANMNIST/")
    model_directory = os.path.join(mnist_dir, "remote_exp/"+lossmetric+"privacy_checkpoints1_1_D"+D+"_order20_xmetricPPANMNIST/")
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.input_size])
    output_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size])
    private_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.private_size])
    prior_tensor = tf.constant(prior, tf.float32, [FLAGS.private_size])
    rawc_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size])
    #load data not necessary for mnist data, formatted as vectors of real values between 0 and 1
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    def get_feed(batch_no, training):
        if training:
            x, c = mnist.train.next_batch(FLAGS.batch_size)
        else:
            x, c = mnist.test.next_batch(FLAGS.batch_size)
        rawc = np.argmax(c, axis=1)
        return {input_tensor: x, output_tensor: x, private_tensor: c[:, :FLAGS.private_size], rawc_tensor: rawc}

    #instantiate model
    with pt.defaults_scope(activation_fn=tf.nn.relu,
                            batch_normalize=True,
                            learned_moments_update_rate=3e-4,
                            variance_epsilon=1e-3,
                            scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("encoder", reuse=False) as scope:
                z = dvibcomp.privacy_encoder(input_tensor, private_tensor)
                encode_params = tf.trainable_variables()
                e_param_len = len(encode_params)
            with tf.variable_scope("decoder", reuse=False) as scope:
                xhat, chat, mean, stddev = dvibcomp.mnist_predictor(z)
                all_params = tf.trainable_variables()
                d_param_len = len(all_params) - e_param_len
   
    # Calculating losses 
    _, KLloss = dvibloss.encoding_cost(xhat, chat, input_tensor, private_tensor, prior_tensor)
    loss2x, loss2c = dvibloss.recon_cost(xhat, chat, input_tensor, private_tensor, softmax=True)
    # Record losses of MI approximation and sibson MI
    #_, _, _, h_c, h_cz, _ = dvibloss.MI_approx(input_tensor, private_tensor, rawc_tensor, xhat, chat, z)
    #I_c_cz = tf.abs(h_c - h_cz)
    # use alpha = 3 first, may be tuned
    sibMI_c_cz = dvibloss.sibsonMI_approx(z, chat, 20)
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
            pub_dist = tf.reduce_mean((xhat - output_tensor)**2)
    with tf.name_scope('sec_prediction'):
        with tf.name_scope('sec_distance'):
            sec_dist = tf.reduce_mean((chat - private_tensor)**2)
            #correct_pred = tf.less(tf.abs(chat - private_tensor), 0.5)
            correct_pred = tf.equal(tf.argmax(chat, axis=1), tf.argmax(private_tensor, axis=1))
            sec_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    pdb.set_trace()
    sess = tf.Session()
    checkpt = tf.train.latest_checkpoint(model_directory)
    saver = tf.train.Saver()
    saver.restore(sess, checkpt)
    print("Restored model from checkpoint %s" % (checkpt))
    x_val = []
    xhat_val = []
   
    feeds = get_feed(FLAGS.test_dataset_size, False)
    x_val.extend(feeds[input_tensor])
    xhat_val.extend(sess.run(xhat, feeds))
    np.savez(os.path.join(model_directory, 'vis_x_xhat'), x=x_val,
                                                  xhat=xhat_val)
    sess.close()
    return



def eval_discrim_checkpt(model_directory):
    sess = tf.Session()
    checkpt = tf.train.latest_checkpoint(model_directory)
    saver.restore(sess, checkpt)
    print("Restored model from checkpoint %s" % (checkpt))
    sess.run(init)



if __name__ == '__main__':
    counts = calc_pc()
    #train_mnist(counts, lossmetric="KL")
    #FLAGS.max_epoch = 2000
    train_mnist(counts)
    #train_mnist_discrim(counts, lossmetric="KL")
    #eval_checkpt(1e-3, "sibMI")

