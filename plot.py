import os
import string
import numpy as np
import tensorflow as tf
import pickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import pdb

def plot_celeba_perf():
    data = np.load('checkpoints/train_perf.npz')
    plt.plot(data['train_loss'], label='Training loss')
    plt.plot(data['train_acc'], label='Training acc')
    plt.plot(np.arange(0,200, 10), data['test_acc'], label='Test acc')
    plt.legend()
    plt.title('Training performance, lr=1e-4')
    plt.savefig('checkpoints/full_perf.png', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_celeba_privacy_perf():
    data = np.load('/home/rxiao/data/celebA/privacy_checkpoints/celeba_trainstats.npz')
    plot_dir = '/home/rxiao/code/dvib/celeba_checkpoints_100'
    #pdb.set_trace()
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    keys = data.keys()

    if 'd_loss_train' in keys and 'd_loss_val' in keys:
        x = np.nonzero(data['d_loss_val'])[0]
        plt.plot(x, data['d_loss_train'][x], label='Training decoding loss')
        plt.plot(x, data['d_loss_val'][x], '-x', label='Validation decoding loss')
        plt.legend()
        plt.title('Decoding loss performance, lr=1e-4')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(plot_dir + '/dec_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'pub_loss_train' in keys and 'pub_loss_val' in keys:
        x = np.nonzero(data['pub_loss_val'])[0]
        plt.plot(x, data['pub_loss_train'][x], '-o', markersize=3, label='Training public distance')
        plt.plot(x, data['pub_loss_val'][x], '-x', markersize=3, label='Validation public distance')
        plt.legend()
        plt.title('Public L2 distance, lr=1e-4')
        plt.xlabel('Epochs')
        plt.ylabel('L2 distance')
        plt.savefig(plot_dir + '/pub_dist.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'e_loss_train' in keys and 'e_loss_val' in keys:
        x = np.nonzero(data['e_loss_train'])[0]
        plt.plot(x, data['e_loss_train'][x], '-o', label='Training encoding loss')
        plt.plot(x, data['e_loss_val'][x], '-x', label='Validation encoding loss')
        plt.legend()
        plt.title('Encoding loss performance, lr=1e-4')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(plot_dir + '/enc_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'sec_loss_train' in keys and 'sec_loss_val' in keys:
        x = np.nonzero(data['sec_loss_train'])[0]
        plt.plot(x, data['sec_loss_train'][x], '-o', label='Training private distance')
        plt.plot(x, data['sec_loss_val'][x], '-x', label='Validation private distance')
        plt.legend()
        plt.title('Private L2 distance, lr=1e-4')
        plt.xlabel('Epochs')
        plt.ylabel('L2 distance')
        plt.savefig(plot_dir + '/sec_dist.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'KL_loss_train' in keys and 'KL_loss_val' in keys:
        x = np.nonzero(data['KL_loss_train'])[0]
        plt.plot(x, data['KL_loss_train'][x], '-o', label='Training KL distance')
        plt.plot(x, data['KL_loss_val'][x], '-x', label='Validation KL distance')
        plt.legend()
        plt.title('KL divergence, lr=1e-4')
        plt.xlabel('Epochs')
        plt.ylabel('KL')
        plt.savefig(plot_dir + '/KLloss.png', bbox_inches='tight')
        plt.show()
        plt.close()

def plot_synth_perf(data, savedir, order=1):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    keys = data.keys()
    #pdb.set_trace()
    
    if 'pub_dist_train' in keys and 'pub_dist_val' in keys:
        plt.plot(data['pub_dist_train'], '-o', markersize=3, label='Training public distance')
        x = np.nonzero(data['pub_dist_val'])[0]
        plt.plot(x, data['pub_dist_val'][x], '-x', markersize=3, label='Validation public distance')
        plt.legend()
        plt.title('Public L2 distance(X), lr=1e-5')
        plt.xlabel('Epochs')
        plt.ylabel('L2 distance')
        plt.savefig(savedir + 'pub_dist.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'sec_dist_train' in keys and 'sec_dist_val' in keys:
        x = np.nonzero(data['sec_dist_train'])[0]
        plt.plot(x, data['sec_dist_train'][x], '-o', label='Training public distance')
        x = np.nonzero(data['sec_dist_val'])[0]
        plt.plot(x, data['sec_dist_val'][x], '-x', label='Validation private distance')
        plt.legend()
        plt.title('Private L2 distance(C), lr=1e-5')
        plt.xlabel('Epochs')
        plt.ylabel('L2 distance')
        plt.savefig(savedir + 'sec_dist.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'KLloss_train' in keys and 'KLloss_val' in keys:
        x = np.nonzero(data['KLloss_train'])[0]
        plt.plot(x, data['KLloss_train'][x], '-o', markersize=3, label='Training KL loss')
        x = np.nonzero(data['KLloss_val'])[0]
        plt.plot(x, data['KLloss_val'][x], '-x', markersize=3, label='Validation KL loss')
        plt.legend()
        plt.title('Training/Validation KL loss, lr=1e-5')
        plt.savefig(savedir + 'KLloss.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'MIloss_train' in keys and 'MIloss_val' in keys:
        x = np.nonzero(data['MIloss_train'])[0]
        plt.plot(x, data['MIloss_train'][x], '-o', markersize=3, label='Training MI approx')
        x = np.nonzero(data['MIloss_val'])[0]
        plt.plot(x, data['MIloss_val'][x], '-x', markersize=3, label='Validation MI approx')
        plt.legend()
        plt.title('Training/Validation MI approximation, lr=1e-5')
        plt.savefig(savedir + 'MIloss.png', bbox_inches='tight')
        plt.show()
        plt.close()

    #pdb.set_trace()
    if 'sibMIloss_train' in keys and 'sibMIloss_val' in keys:
        x = np.nonzero(data['KLloss_train'])[0]
        plt.plot(x, data['KLloss_train'][x], '--', markersize=3, label='Training KL loss')
        x = np.nonzero(data['KLloss_val'])[0]
        plt.plot(x, data['KLloss_val'][x], '-o', markersize=3, label='Validation KL loss')
        #x = np.nonzero(data['sibMIloss_train'])[0]
        #plt.plot(x, data['MIloss_train'][x], '-', markersize=1, label='Training MI approx')
        #x = np.nonzero(data['sibMIloss_train'])[0]
        #plt.plot(x, data['MIloss_val'][x], '-', markersize=1, label='Validation MI approx')
        x = np.nonzero(data['sibMIloss_train'])[0]
        plt.plot(x, data['sibMIloss_train'][x], '-d', markersize=3, label='Training sibson MI approx')
        x = np.nonzero(data['sibMIloss_val'])[0]
        plt.plot(x, data['sibMIloss_val'][x], '-', markersize=3, label='Validation sibson MI approx')
        plt.legend()
        plt.title('Training/Validation sibson MI approximation, order '+str(order)+', lr=1e-5')
        plt.savefig(savedir + 'sibMIloss.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()

    if 'sec_acc_train' in keys and 'sec_acc_val' in keys:
        x = np.nonzero(data['sec_acc_train'])[0]
        plt.plot(x, data['sec_acc_train'][x], '-d', markersize=3, label='Training private accuracy')
        x = np.nonzero(data['sec_acc_val'])[0]
        plt.plot(x, data['sec_acc_val'][x], '--', markersize=3, label='Validation private accuracy')
        if 'pub_acc_train' in keys and 'pub_acc_val' in keys:
            x = np.nonzero(data['pub_acc_train'])[0]
            plt.plot(x, data['pub_acc_train'][x], '-^', markersize=3, label='Training public accuracy')
            x = np.nonzero(data['pub_acc_val'])[0]
            plt.plot(x, data['pub_acc_val'][x], '--', markersize=3, label='Validation public accuracy')
            plt.legend()
            plt.title('Training/Validation public/private accuracy, lr=1e-5')
            plt.savefig(savedir + 'pub_sec_acc.png', bbox_inches='tight')
        else:
            plt.legend()
            plt.title('Training/Validation private accuracy, lr=1e-5')
            plt.savefig(savedir + 'sec_acc.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'e_loss_train' in keys and 'e_loss_val' in keys:
        x = np.nonzero(data['e_loss_train'])[0]
        plt.plot(x, data['e_loss_train'][x], '-o', label='Training encoding loss')
        x = np.nonzero(data['e_loss_val'])[0]
        plt.plot(x, data['e_loss_val'][x], '-x', label='Validation encoding loss')
        plt.legend()
        if 'g_loss_train' in keys and 'g_loss_val' in keys:
            plt.title('Encoding loss: Gen loss*encode_coef+KL, lr=1e-5')
        else:
            plt.title('Encoding loss performance, lr=1e-5')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(savedir + 'enc_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'g_loss_train' in keys and 'g_loss_val' in keys:
        x = np.nonzero(data['g_loss_train'])[0]
        plt.plot(x, data['g_loss_train'][x], '-o', markersize=3, label='Training loss')
        x = np.nonzero(data['g_loss_val'])[0]
        plt.plot(x, data['g_loss_val'][x], '-x', markersize=3, label='Validation loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Decoding loss: Gen loss*decode_coef+cross-entropy, lr=1e-5')
        plt.savefig(savedir + 'g_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'd_loss_train' in keys and 'd_loss_val' in keys:
        plt.plot(data['d_loss_train'], label='Training decoding loss')
        x = np.nonzero(data['d_loss_val'])[0]
        plt.plot(x, data['d_loss_val'][x], '-x', label='Validation decoding loss')
        plt.legend()
        if 'g_loss_train' in keys and 'g_loss_val' in keys:
            plt.title('Discriminator loss, lr=1e-5')
        else:
            plt.title('Decoding loss performance, lr=1e-5')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(savedir + 'dec_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()
    # For FERG experiments with loss on y
    if 'loss2y_train' in keys and 'loss2y_val' in keys:
        x = np.nonzero(data['loss2y_train'])[0]
        plt.plot(x, data['loss2y_train'][x], '-o', markersize=3, label='Training loss')
        x = np.nonzero(data['loss2y_val'])[0]
        plt.plot(x, data['loss2y_val'][x], '-x', markersize=3, label='Validation loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss on Y(Cross-entropy), lr=1e-5')
        plt.savefig(savedir + 'loss2y.png', bbox_inches='tight')
        plt.show()
        plt.close()


def plot_tmp():
    pdb.set_trace()
    data = np.load('/home/rxiao/data/synthetic/privacy_checkpoints/synth_trainstats.npz')
    x = np.nonzero(data['KLloss_train'])[0]
    plt.plot(x, data['KLloss_train'][x], '-o', markersize=3, label='Training KL loss')
    x = np.nonzero(data['KLloss_val'])[0]
    plt.plot(x, data['KLloss_val'][x], '-x', markersize=3, label='Validation KL loss')
    plt.legend()
    plt.title('Training/Validation KL loss, lr=1e-5')
    plt.savefig('synth/KLloss.png', bbox_inches='tight')
    plt.show()
    #plt.close()

def plot_synthsib_diff_coefs(datadir='/home/rxiao/data/synthetic/', savedir='synthsib_diff_coefs/'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    listdir = os.listdir(datadir)
    if "ferg256.npz" in listdir:
        listdir.remove("ferg256.npz")
    if "1d2gaussian.npz" in listdir:
        listdir.remove("1d2gaussian.npz")
    KL_loss_train=[]
    KL_loss_val=[]
    sibMI_loss_train=[]
    sibMI_loss_val=[]
    pub_dist_train=[]
    pub_dist_val=[]
    sec_dist_train=[]
    sec_dist_val=[]
    sec_acc_train=[]
    sec_acc_val=[]
    x = []
    for dir in  listdir:
        tmpentry = dir.split('privacy_checkpoints')
        params = tmpentry[1].split("_")
        if tmpentry[0]=="sibMI_" and len(params)==3 and params[0]=="0.001" and params[1]=="1":
            encode_coef = float(params[0])
            decode_coef = float(params[1])
            order = float(params[2])
            x.append(order)
            data = np.load(datadir+dir+'/synth_trainstats.npz')
            sibMI_loss_train.append(data['e_loss_train'][-1])
            sibMI_loss_val.append(data['e_loss_val'][-1])
            KL_loss_train.append(data['KLloss_train'][-1])
            KL_loss_val.append(data['KLloss_val'][-1])
            sec_acc_train.append(data['sec_acc_train'][-1])
            sec_acc_val.append(data['sec_acc_val'][-1])
            #pdb.set_trace()
     
    x = np.array(x)
    #x = np.array([float(entry.split('privacy_checkpoints')[-1]) for entry in listdir])
    indices = np.argsort(x)
    #plt.semilogx(x[indices], np.array(KL_loss_train)[indices], '-x', markersize=2, label='Training KL loss')
    #plt.semilogx(x[indices], np.array(KL_loss_val)[indices], '-o', markersize=2, label='Validation KL loss')
    plt.plot(x[indices], np.array(KL_loss_train)[indices], '-x', markersize=2, label='Training KL loss')
    plt.plot(x[indices], np.array(KL_loss_val)[indices], '-o', markersize=2, label='Validation KL loss')
    plt.plot(x[indices], np.array(sibMI_loss_train)[indices], '-x', markersize=2, label='Training Sibson MI loss')
    plt.plot(x[indices], np.array(sibMI_loss_val)[indices], '-o', markersize=2, label='Validation Sibson MI loss')
    plt.legend()
    plt.title('Training/Validation KL/Sibson MI, encoding coeff 0.001')
    plt.xlabel('Order of Sibson MI')
    plt.ylabel('Loss')
    plt.savefig(savedir + 'sibMI_KL_loss.png', bbox_inches='tight')
    plt.show()
    plt.close()

    #pdb.set_trace()
    #plt.semilogx(x[indices], np.array(sec_acc_train)[indices], '-x', markersize=2, label='Training private accuracy')
    #plt.semilogx(x[indices], np.array(sec_acc_val)[indices], '-o', markersize=2, label='Validation priavte accuracy')
    plt.plot(x[indices], np.array(sec_acc_train)[indices], '-x', markersize=1, label='Training private accuracy')
    plt.plot(x[indices], np.array(sec_acc_val)[indices], '-o', markersize=1, label='Validation private accuracy')
    plt.legend()
    plt.title('Training/Validation private accuracy, encoding coeff 0.001')
    plt.xlabel('Order of Sibson MI')
    plt.ylabel('Accuracy %')
    plt.savefig(savedir+'/sec_acc.png', bbox_inches='tight')
    plt.show()
    plt.close()

    return


def plot_ferg_diff_coefs(datadir='/home/rxiao/data/ferg/', savedir='ferg_diff_coefs/'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    listdir = os.listdir(datadir)
    if "ferg256.npz" in listdir:
        listdir.remove("ferg256.npz")
    enc_loss_train=[]
    enc_loss_val=[]
    dec_loss_train=[]
    dec_loss_val=[]
    KL_loss_train=[]
    KL_loss_val=[]
    pub_dist_train=[]
    pub_dist_val=[]
    sec_dist_train=[]
    sec_dist_val=[]
    pub_acc_train=[]
    pub_acc_val=[]
    sec_acc_train=[]
    sec_acc_val=[]
    for dir in  listdir:
        data = np.load(datadir+dir+'/ferg_trainstats.npz')
        enc_loss_train.extend([data['e_loss_train'][-1]])
        enc_loss_val.extend([data['e_loss_val'][-1]])
        dec_loss_train.extend([data['d_loss_train'][-1]])
        dec_loss_val.extend([data['d_loss_val'][-1]])
        KL_loss_train.extend([data['KLloss_train'][-1]])
        KL_loss_val.extend([data['KLloss_val'][-1]])
        pub_dist_train.extend([data['pub_dist_train'][-1]])
        pub_dist_val.extend([data['pub_dist_val'][-1]])
        sec_dist_train.extend([data['sec_dist_train'][-1]])
        sec_dist_val.extend([data['sec_dist_val'][-1]])
        pub_acc_train.extend([data['pub_acc_train'][-1]])
        pub_acc_val.extend([data['pub_acc_val'][-1]])
        sec_acc_train.extend([data['sec_acc_train'][-1]])
        sec_acc_val.extend([data['sec_acc_val'][-1]])

    pdb.set_trace()
    x = np.array([float(entry.split('privacy_checkpoints')[-1]) for entry in listdir])
    indices = np.argsort(x)
    pdb.set_trace()
    #plt.semilogx(x[indices], np.array(enc_loss_train)[indices], '-x', markersize=2, label='Training encoding loss')
    #plt.semilogx(x[indices], np.array(enc_loss_val)[indices], '-o', markersize=2, label='Validation encoding loss')
    plt.plot(x[indices], np.array(enc_loss_train)[indices], '-x', markersize=2, label='Training encoding loss')
    plt.plot(x[indices], np.array(enc_loss_val)[indices], '-o', markersize=2, label='Validation encoding loss')
    plt.legend()
    plt.title('Training/Validation encoding loss, lr=1e-5')
    plt.xlabel('Encoding coefficient')
    plt.ylabel('Loss')
    plt.savefig(savedir + 'enc_loss.png', bbox_inches='tight')
    plt.show()
    plt.close()

    #plt.semilogx(x[indices], np.array(dec_loss_train)[indices], '-x', markersize=2, label='Training decoding loss')
    #plt.semilogx(x[indices], np.array(dec_loss_val)[indices], '-o', markersize=2, label='Validation decoding loss')
    plt.plot(x[indices], np.array(dec_loss_train)[indices], '-x', markersize=2, label='Training decoding loss')
    plt.plot(x[indices], np.array(dec_loss_val)[indices], '-o', markersize=2, label='Validation decoding loss')
    plt.legend()
    plt.title('Training/Validation decoding loss, lr=1e-5')
    plt.xlabel('Encoding coefficient')
    plt.ylabel('Loss')
    plt.savefig(savedir + 'dec_loss.png', bbox_inches='tight')
    plt.show()
    plt.close()

    #plt.semilogx(x[indices], np.array(KL_loss_train)[indices], '-x', markersize=2, label='Training KL loss')
    #plt.semilogx(x[indices], np.array(KL_loss_val)[indices], '-o', markersize=2, label='Validation KL loss')
    plt.plot(x[indices], np.array(KL_loss_train)[indices], '-x', markersize=2, label='Training KL loss')
    plt.plot(x[indices], np.array(KL_loss_val)[indices], '-o', markersize=2, label='Validation KL loss')
    plt.legend()
    plt.title('Training/Validation KL loss, lr=1e-5')
    plt.xlabel('Encoding coefficient')
    plt.ylabel('Loss')
    plt.savefig(savedir + 'KL_loss.png', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(x[indices], np.array(pub_dist_train)[indices], '-x', markersize=2, label='Training public distance')
    plt.plot(x[indices], np.array(pub_dist_val)[indices], '-o', markersize=2, label='Validation public distance')
    plt.plot(x[indices], np.array(sec_dist_train)[indices], '-x', markersize=2, label='Training private distance')
    plt.plot(x[indices], np.array(sec_dist_val)[indices], '-o', markersize=2, label='Validation private distance')
    plt.legend()
    plt.title('Training/Validation public/private distance, lr=1e-5')
    plt.xlabel('Encoding coefficient')
    plt.ylabel('L2 distance')
    plt.savefig(savedir + 'dist.png', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(x[indices], np.array(pub_acc_train)[indices], '-x', markersize=1, label='Training public accuracy')
    plt.plot(x[indices], np.array(pub_acc_val)[indices], '-o', markersize=1, label='Validation public accuracy')
    plt.plot(x[indices], np.array(sec_acc_train)[indices], '-x', markersize=1, label='Training private accuracy')
    plt.plot(x[indices], np.array(sec_acc_val)[indices], '-o', markersize=1, label='Validation private accuracy')
    plt.legend()
    plt.title('Training/Validation public/private accuracy, lr=1e-5')
    plt.xlabel('Encoding coefficient')
    plt.ylabel('Accuracy %')
    plt.savefig(savedir+'/pub_sec_acc.png', bbox_inches='tight')
    plt.show()
    plt.close()

    return

def plot_mnist_diff_coefs(datadir='/home/rxiao/data/mnist/remote_exp/', savedir='mnist_diff_coefs/'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    listdir = os.listdir(datadir)
    # Remove non-directory elements
    if "ferg256.npz" in listdir:
        listdir.remove("ferg256.npz")
    removelist = []
    for elem in listdir:
        if '.gz' in elem:
            removelist.append(elem)
    for elem in removelist:
        listdir.remove(elem)
    
    # Parse hyperparameters
    hyperparams = []
    for dir in listdir:
        metric = dir.split('privacy_checkpoints')[0]
        if metric=='sibMI':
            params = dir.split('privacy_checkpoints')[1].split('_')
            encode_coef = float(params[0])
            decode_coef = float(params[1])
            order = float(params[2])
            hyperparams.append({'dir': dir,
                                'metric': metric,
                                'encode_coef': encode_coef,
                                'decode_coef': decode_coef,
                                'order': order})

    enc_loss_train=[]
    enc_loss_val=[]
    dec_loss_train=[]
    dec_loss_val=[]
    KL_loss_train=[]
    KL_loss_val=[]
    sibMI_loss_train=[]
    sibMI_loss_val=[]
    pub_dist_train=[]
    pub_dist_val=[]
    sec_dist_train=[]
    sec_dist_val=[]
    sec_acc_train=[]
    sec_acc_val=[]
    x = []
    keys = {}
    for params in hyperparams:
        data = np.load(datadir+params['dir']+'/synth_trainstats.npz')
        keys = data.keys()
        enc_loss_train.extend([data['e_loss_train'][-1]])
        enc_loss_val.extend([data['e_loss_val'][-1]])
        dec_loss_train.extend([data['d_loss_train'][-1]])
        dec_loss_val.extend([data['d_loss_val'][-1]])
        KL_loss_train.extend([data['KLloss_train'][-1]])
        KL_loss_val.extend([data['KLloss_val'][-1]])
        sibMI_loss_train.extend([data['KLloss_train'][-1]])
        sibMI_loss_val.extend([data['KLloss_val'][-1]])
        pub_dist_train.extend([data['pub_dist_train'][-1]])
        pub_dist_val.extend([data['pub_dist_val'][-1]])
        sec_dist_train.extend([data['sec_dist_train'][-1]])
        sec_dist_val.extend([data['sec_dist_val'][-1]])
        sec_acc_train.extend([data['sec_acc_train'][-1]])
        sec_acc_val.extend([data['sec_acc_val'][-1]])
        x.append(params['encode_coef'])

    indices = np.argsort(x)
    plot_enc_loss_coef(keys, indices, x, enc_loss_train, enc_loss_val, savedir)
    plot_dec_loss_coef(keys, indices, x, dec_loss_train, dec_loss_val, savedir)
    plot_KL_loss_coef(keys, indices, x, KL_loss_train, KL_loss_val, savedir)
    plot_MI_loss_coef(keys, indices, x, MI_loss_train, MI_loss_val, savedir)
    plot_sibMI_loss_coef(keys, indices, x, sibMI_loss_train, sibMI_loss_val, savedir)
    plot_dist_coef(keys, indices, x, pub_dist_train, pub_dist_val, sec_dist_train, sec_dist_val, savedir)
    plot_sec_acc_coef(keys, indices, x, sec_acc_train, sec_acc_val, savedir)
    return

def plot_enc_loss_coef(keys, indices, x, enc_loss_train, enc_loss_val, savedir):
    x = np.array(x)
    if 'e_loss_train' in keys and 'e_loss_val' in keys:
        plt.plot(x[indices], np.array(enc_loss_train)[indices], '-x', markersize=2, label='Training encoding loss')
        plt.plot(x[indices], np.array(enc_loss_val)[indices], '-o', markersize=2, label='Validation encoding loss')
        plt.legend()
        plt.title('Training/Validation encoding loss, lr=1e-5')
        plt.xlabel('Encoding coefficient')
        plt.ylabel('Loss')
        plt.savefig(savedir + 'enc_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()
    return

def plot_dec_loss_coef(keys, indices, x, dec_loss_train, dec_loss_val, savedir):
    x = np.array(x)
    if 'd_loss_train' in keys and 'd_loss_val' in keys:
        plt.plot(x[indices], np.array(dec_loss_train)[indices], '-x', markersize=2, label='Training decoding loss')
        plt.plot(x[indices], np.array(dec_loss_val)[indices], '-o', markersize=2, label='Validation decoding loss')
        plt.legend()
        plt.title('Training/Validation decoding loss')
        plt.xlabel('Encoding coefficient')
        plt.ylabel('Loss')
        plt.savefig(savedir + 'dec_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()
    return

def plot_KL_loss_coef(keys, indices, x, KL_loss_train, KL_loss_val, savedir):
    x = np.array(x)
    if 'KL_loss_train' in keys and 'KL_loss_val' in keys:
        plt.plot(x[indices], np.array(KL_loss_train)[indices], '-x', markersize=2, label='Training KL loss')
        plt.plot(x[indices], np.array(KL_loss_val)[indices], '-o', markersize=2, label='Validation KL loss')
        plt.legend()
        plt.title('Training/Validation KL loss')
        plt.xlabel('Encoding coefficient')
        plt.ylabel('Loss')
        plt.savefig(savedir + 'KL_loss.png', bbox_inches='tight')
        plt.show()
        plt.close()
    return

def plot_MI_loss_coef(keys, indices, x, MI_loss_train, MI_loss_val, savedir):
    x = np.array(x)
    if 'MIloss_train' in keys and 'MIloss_val' in keys:
        plt.plot(x[indices], np.array(MI_loss_train)[indices], '-o', markersize=3, label='Training MI approx')
        plt.plot(x[indices], np.array(MI_loss_val)[indices], '-x', markersize=3, label='Validation MI approx')
        plt.legend()
        plt.title('Training/Validation MI approximation')
        plt.savefig(savedir + 'MIloss.png', bbox_inches='tight')
        plt.show()
        plt.close()
    return

def plot_sibMI_loss_coef(keys, indices, x, sibMI_loss_train, sibMI_loss_val, savedir, order):
    x = np.array(x)
    if 'sibMIloss_train' in keys and 'sibMIloss_val' in keys:
        plt.plot(x[indices], np.array(sibMI_loss_train)[indices], '-d', markersize=3, label='Training sibson MI approx')
        plt.plot(x[indices], np.array(sibMI_loss_val)[indices], '-', markersize=3, label='Validation sibson MI approx')
        plt.legend()
        plt.title('Training/Validation sibson MI approximation, order '+str(order)+', lr=1e-5')
        plt.xlabel('Encoding coefficient')
        plt.ylabel('Sibson MI')
        plt.savefig(savedir + 'sibMIloss.png', bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()

def plot_dist_coef(keys, indices, x, pub_dist_train, pub_dist_val, sec_dist_train, sec_dist_val, savedir):
    x = np.array(x)
    if 'pub_dist_train' in keys and 'pub_dist_val' in keys:
        plt.plot(x[indices], np.array(pub_dist_train)[indices], '-x', markersize=2, label='Training public distance')
        plt.plot(x[indices], np.array(pub_dist_val)[indices], '-o', markersize=2, label='Validation public distance')
        plt.plot(x[indices], np.array(sec_dist_train)[indices], '-x', markersize=2, label='Training private distance')
        plt.plot(x[indices], np.array(sec_dist_val)[indices], '-o', markersize=2, label='Validation private distance')
        plt.legend()
        plt.title('Training/Validation public/private distance')
        plt.xlabel('Encoding coefficient')
        plt.ylabel('L2 distance')
        plt.savefig(savedir + 'dist.png', bbox_inches='tight')
        plt.show()
        plt.close()
    return

def plot_pub_sec_acc_coef(keys, indices, x, pub_acc_train, pub_acc_val, sec_acc_train, sec_acc_val, savedir):
    x = np.array(x)
    sec_acc_train = np.array(sec_acc_train)
    sec_acc_val = np.array(sec_acc_val)
    pub_acc_train = np.array(pub_acc_train)
    pub_acc_val = np.array(pub_acc_val)
    plt.plot(x[indices], sec_acc_train[indices], '-d', markersize=3, label='Training private accuracy')
    plt.plot(x[indices], sec_acc_val[indices], '--', markersize=3, label='Validation private accuracy')
    plt.plot(x[indices], pub_acc_train[indices], '-^', markersize=3, label='Training public accuracy')
    plt.plot(x[indices], pub_acc_val[indices], '--', markersize=3, label='Validation public accuracy')
    plt.legend()
    plt.title('Training/Validation public/private accuracy')
    plt.xlabel('Encoding coefficient')
    plt.ylabel('Accuracy')
    plt.savefig(savedir + 'pub_sec_acc.png', bbox_inches='tight', dpi=1000)
    plt.show()
    plt.close()
    return

def plot_sec_acc_coef(keys, indices, x, sec_acc_train, sec_acc_val, savedir):
    x = np.array(x)
    sec_acc_train = np.array(sec_acc_train)
    sec_acc_val = np.array(sec_acc_val)
    plt.semilogx(x[indices], sec_acc_train[indices], '-d', markersize=3, label='Training private accuracy')
    plt.semilogx(x[indices], sec_acc_val[indices], '--', markersize=3, label='Validation private accuracy')
    plt.legend()
    plt.xlabel('Encoding coefficient')
    plt.ylabel('Accuracy')
    plt.title('Training/Validation private accuracy')
    plt.savefig(savedir + 'sec_acc.png', bbox_inches='tight')
    plt.show()
    plt.close()
    return

def plot_x_scatter(savedir, savename, x, y, legend, xlabel, ylabel):
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y)
    plt.title(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savedir + savename, bbox_inches='tight')
    plt.show()
    plt.close()
    return

def plot_x_scatter_with_theoretic(savedir, savename, x, y, theox, theoy, legend, xlabel, ylabel, label1="Experiment", label2="Theory"):
    x = np.array(x)
    y = np.array(y)
    theox = np.array(theox)
    theoy = np.array(theoy)
    index = np.argsort(theox)
    pdb.set_trace()
    plt.plot(theox[index][:], theoy[index][:], '-d', markersize=4, label=label2)
    index = np.argsort(x)
    plt.plot(x[index][:], y[index][:], '-o', markersize=4, label=label1)
    plt.title(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(savedir + savename, bbox_inches='tight')
    plt.show()
    plt.close()
    return

def plot_x_scatter_with_theoretic_compare(savedir, savename, x, y, theox, theoy, compx, compy, comptheox, comptheoy,
        legend, xlabel, ylabel, label1="Experiment", label2="Theory", label3="KL", label4="SibMI"):
    x = np.array(x)
    y = np.array(y)
    theox = np.array(theox)
    theoy = np.array(theoy)
    index = np.argsort(theox)
    plt.plot(theox[index], theoy[index], '-d', markersize=4, label=label2)
    plt.plot(x[index], y[index], '-o', markersize=4, label=label1)
    index = np.argsort(comptheox)
    comptheox = np.array(comptheox)
    comptheoy = np.array(comptheoy)
    compx = np.array(compx)
    compy = np.array(compy)
    plt.plot(comptheox[index], comptheoy[index], '-d', markersize=4, label=label3)
    plt.plot(compx[index], compy[index], '-o', markersize=4, label=label4)
    plt.title(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(savedir + savename, bbox_inches='tight')
    plt.show()
    plt.close()
    return

def plot_mnist_x_xhat(data, savedir, encode_coef, rows, cols):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
 
    xval = data['x']
    xhatval = data['xhat']
    n,xdim = xval.shape

    # reshape data if mnist digits
    if xdim==784:
        xval = xval.reshape(n, 28, 28)
        xhatval = xhatval.reshape(n, 28, 28)
    fig = plt.figure(figsize=(8,8))
    indices = np.random.randint(0, n, size=rows*cols)
    for i in xrange(1, rows*cols+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(xval[indices[i-1]])
    plt.savefig(savedir+'/xvis'+str(encode_coef)+'.png', bbox_inches='tight')
    plt.close()
    
    fig = plt.figure(figsize=(8,8))
    for i in xrange(1, rows*cols+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(xhatval[indices[i-1]])
    plt.savefig(savedir+'/xhatvis'+str(encode_coef)+'.png', bbox_inches='tight')
    plt.close()
    print('Finished visualizations of x, xhat, saved to '+ savedir+'/xhatvis'+str(encode_coef)+'.png')
    return

def plot_ferg_x_xhat(data, indices, savedir, rows, cols=2):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
 
    #indices = np.random.randint(0, n, size=rows*cols)  # generate random indices for images/recons
    xval = data['x']
    xhatval = data['xhat']
    n,xdim = xval.shape

    # reshape data if mnist digits
    if xdim == 784:
        xval = xval.reshape(n, 28, 28)
        xhatval = xhatval.reshape(n, 28, 28)
    if xdim == 2500:
        xval = xval.reshape(n, 50, 50)
        xhatval = xhatval.reshape(n, 50, 50)
    fig = plt.figure(figsize=(6,6))
    for i in xrange(1, rows*cols+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(xval[indices[i-1]])
    plt.savefig(savedir+'/xvis.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(6,6))
    for i in xrange(1, rows*cols+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(xhatval[indices[i-1]])
    plt.savefig(savedir+'/xhatvis.png', bbox_inches='tight')
    plt.show()
    plt.close()
    print('Finished visualizations of x, xhat, saved to '+ savedir+'xhatvis.png')
    
    return

def plot_ferg_Dy_experiments(savedir='/home/rxiao/code/dvib/ferg/Dy/'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #construct the data from plots, use final resting accuracies for each experiment run with the Dy budget
    Dy = [0.6, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74]
    final_pub_acc = [0.979, 0.319, 0.473, 0.609, 0.609, 0.729, 0.741]
    final_sec_acc = [0.457, 0.260, 0.285, 0.339, 0.341, 0.387, 0.384]
    pub_sec_expdy_pairs_sibMI = [[0.865, 0.379, 0.656],
                     [0.979, 0.457, 0.641],
                     [0.319, 0.260, 0.736],
                     [0.472, 0.281, 0.714],
                     [0.455, 0.283, 0.717],
                     [0.473, 0.285, 0.714],
                     [0.472, 0.280, 0.714],
                     [0.609, 0.339, 0.694],
                     [0.609, 0.341, 0.694],
                     [0.473, 0.286, 0.714],
                     [0.729, 0.367, 0.675],
                     [0.741, 0.364, 0.676]]
    pub_sec_expdy_pairs_KL = [#new experiments
                              [0.741, 0.401, 0.665],
                              [0.522, 0.369, 0.702],
                              [0.862, 0.415, 0.653],
                              # old experiments
                              [0.604, 0.387, 0.695],
                              [0.473, 0.309, 0.714]
                             ]

    # Ignoring the first run as the budget was not satisfied by the model
    Dy = Dy[1:]
    final_pub_acc = final_pub_acc[1:]
    final_sec_acc = final_sec_acc[1:]
    #plot_x_scatter_with_theoretic(savedir, 'Dy_exp_final_accs', Dy, final_pub_acc, Dy, final_sec_acc, 'Final accuracies vs distortion budget on regular Y', 'Y distortion budget', 'Task accuracy', 'Regular task', 'Private task')
    #plot_x_scatter(savedir, 'Dy_exp_pairs', final_pub_acc, final_sec_acc, 'Regular task accuracy vs private task', 'Regular task accuracy', 'Private task accuracy')

    #pdb.set_trace()
    pub_sec_expdy_pairs_sibMI = np.array(pub_sec_expdy_pairs_sibMI)
    pub_sec_expdy_pairs_KL = np.array(pub_sec_expdy_pairs_KL)
    #plot_x_scatter_with_theoretic(savedir, 'expDy_exp_final_accs', pub_sec_expdy_pairs[:, 2], pub_sec_expdy_pairs[:, 0], pub_sec_expdy_pairs[:, 2], pub_sec_expdy_pairs[:, 1], 'Final accuracies vs distortion budget on regular Y', 'Loss on Y(Cross-entropy)', 'Task accuracy', 'Regular task', 'Private task')
    plot_x_scatter_with_theoretic_compare(savedir, 'expDy_exp_final_accs', pub_sec_expdy_pairs_sibMI[:, 2], pub_sec_expdy_pairs_sibMI[:, 0], pub_sec_expdy_pairs_sibMI[:, 2], pub_sec_expdy_pairs_sibMI[:, 1], pub_sec_expdy_pairs_KL[:, 2], pub_sec_expdy_pairs_KL[:, 0], pub_sec_expdy_pairs_KL[:, 2], pub_sec_expdy_pairs_KL[:, 1], 'Final accuracies vs distortion budget on regular Y', 'Loss on Y(Cross-entropy)', 'Task accuracy', 'Regular task(Sibson MI)', 'Private task(Sibson MI)', 'Private task(KL)', 'Regular task(KL)')
    plot_x_scatter_with_theoretic(savedir, 'expDy_exp_pairs', pub_sec_expdy_pairs_sibMI[:, 1], pub_sec_expdy_pairs_sibMI[:, 0], pub_sec_expdy_pairs_KL[:, 1], pub_sec_expdy_pairs_KL[:, 0], 'Final accuracies private vs regular task', 'Private Task accuracy', 'Regular Task accuracy', 'Sibson MI', 'KL')
    #plot_x_scatter(savedir, 'expDy_exp_pairs', pub_sec_expdy_pairs[:, 0], pub_sec_expdy_pairs[:, 1], 'Regular task accuracy vs private task', 'Regular task accuracy', 'Private task accuracy')
    return

def plot_synth_compare_affine(savedir='synth/compare_affine'):
    """ Plot the table of synthetic data for affine and affine with noise compared with GAP
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # recorded MAP accuracies for various encoders
    distbudget = range(1, 7)
    gapacc = [0.9742, 0.9169, 0.8633, 0.8123, 0.7545, 0.7122]
    affinedist = [0.738, 1.340, 2.904, 3.174, 3.750, 4.570]
    affineacc = [0.980, 0.965, 0.900, 0.882, 0.850, 0.800]
    noisyaffinedist = [0.936, 1.56, 2.31, 3.08, 4.80, 5.38]
    noisyaffineacc = [0.975, 0.951, 0.926, 0.885, 0.784, 0.741]
    NNdist = [0.8665, 1.7588, 2.185, 2.2358, 3.0467, 4.4257]
    NNacc = [0.9745, 0.9283, 0.8218, 0.6486, 0.5600, 0.5377]
    NNKLdist = [1.67, 2.62, 3.64, 4.02, 4.60, 5.05, 5.31]
    NNKLacc = [0.942, 0.921, 0.868, 0.778, 0.735, 0.724, 0.629]
    #load theoretical data for affine
    theorydata = np.load("synth/theoretic_upper_bound_alpha5.npz")
    # plot the gap, theoretical, affine, noisyaffine, and NN
    plt.plot(distbudget, gapacc, '-.', label="GAP accuracy (affine)")
    plt.plot(affinedist, affineacc, '-x', label="Sibson MI accuracy (affine)")
    plt.plot(noisyaffinedist, noisyaffineacc, '-d', label="Sibson MI accuracy (noisy affine)")
    plt.plot(NNdist, NNacc, '-o', label="Sibson MI accuracy (NN)")
    plt.plot(NNKLdist, NNKLacc, '-x', label="MI accuracy (NN)")
    plt.plot(theorydata['Ds'], theorydata['phis'], label="Theoretical accuracy (affine)")
    plt.xlabel("Distortion")
    plt.ylabel("Private accuracy")
    plt.legend()
    plt.savefig(savedir+'/compare_priv_acc.png', bbox_inches='tight')
    plt.show()
    plt.close()
    return

def report_ferg_dy_experiments(savedir="ferg/", loaddir = "ferg/Dy/"):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    pdb.set_trace()
    if os.path.isfile(os.path.join(loaddir, "reported_Dy_expdata.npz")):
        data = np.load(os.path.join(loaddir, "reported_Dy_expdata.npz"))
        sibpubaccs = data['sibpubaccs']
        sibsecaccs = data['sibsecaccs']
        sibloss2ys = data['sibloss2ys']
        KLpubaccs = data['KLpubaccs']
        KLsecaccs = data['KLsecaccs']
        KLloss2ys = data['KLloss2ys']
    else:
        rangedy = [0.2, 0.3, 0.4, 0.5, 0.6]
        rangedy = np.arange(0.2, 2, step=0.1)
        sibpubaccs = []
        sibsecaccs = []
        sibloss2ys = []
        KLpubaccs = []
        KLsecaccs = []
        KLloss2ys = []
        for i in rangedy:
            sibmodeldir = os.path.join("/cad2/ece521s/cuda_libs/data/ferg/", "sibMIprivacy_checkpoints_Dy_"+str(i)+"_1_1_40/")
            if os.path.exists(sibmodeldir) and os.path.isfile(sibmodeldir+"ferg_trainstats.npz"):
                sibdata = np.load(sibmodeldir+"ferg_trainstats.npz")
                sibpubaccval = sibdata['pub_acc_val'][-1]
                sibsecaccval = sibdata['sec_acc_val'][-1]
                sibloss2yval = sibdata['loss2y_val'][-1]
                if sibloss2yval < i:
                    sibloss2ys.append(sibloss2yval)
                    sibpubaccs.append(sibpubaccval)
                    sibsecaccs.append(sibsecaccval)
            KLmodeldir = os.path.join("/cad2/ece521s/cuda_libs/data/ferg/", "KLprivacy_checkpoints_Dy_"+str(i)+"_1_1_40/")
            if os.path.exists(KLmodeldir) and os.path.isfile(KLmodeldir+"ferg_trainstats.npz"):
                KLdata = np.load(KLmodeldir+"ferg_trainstats.npz")
                KLpubaccval = KLdata['pub_acc_val'][-1]
                KLsecaccval = KLdata['sec_acc_val'][-1]
                KLloss2yval = KLdata['loss2y_val'][-1]
                if KLloss2yval < i:
                    KLloss2ys.append(KLloss2yval)
                    KLpubaccs.append(KLpubaccval)
                    KLsecaccs.append(KLsecaccval)
        np.savez(os.path.join(savedir, "reported_Dy_expdata.npz"), sibpubaccs = sibpubaccs,
            sibsecaccs = sibsecaccs,
            sibloss2ys = sibloss2ys,
            KLpubaccs = KLpubaccs,
            KLsecaccs = KLsecaccs,
            KLloss2ys = KLloss2ys)
    
    # stored data from runs of models on different orders of alpha
    sibpubaccs_order2 = np.array([0.9494, 0.8804, 0.8007, 0.7256, 0.6281, 0.5285, 0.4456, 0.3612, 0.2806])
    sibsecaccs_order2 = np.array([0.3138, 0.3022, 0.3029, 0.2987, 0.2866, 0.2844, 0.2774, 0.2708, 0.2534])
    sibloss2ys_order2 = np.arange(0.2, 2, 0.2)
    
    sibpubaccs_order10 = np.array([0.9500, 0.8806, 0.7955, 0.7184, 0.6240, 0.5363, 0.4420, 0.3556, 0.2706])
    sibsecaccs_order10 = np.array([0.3042, 0.2931, 0.2889, 0.2794, 0.2823, 0.2787, 0.2670, 0.2724, 0.2490])
    sibloss2ys_order10 = np.arange(0.2, 2, 0.2)

    plt.plot(KLloss2ys, KLpubaccs, '-o', markersize=1, label="Public task (MI)")
    plt.plot(KLloss2ys, KLsecaccs, '-d', markersize=2, label="Private task (MI)")
    plt.plot(sibloss2ys, sibpubaccs, '-o', markersize=1, label="Public task (Sibson MI order 40)")
    plt.plot(sibloss2ys, sibsecaccs, '--', markersize=2, label="Private task (Sibson MI order 40)")
    plt.plot(sibloss2ys_order2, sibpubaccs_order2, '-o', markersize=1, label="Public task (Sibson MI order 2)")
    plt.plot(sibloss2ys_order2, sibsecaccs_order2, '-*', markersize=2, label="Private task (Sibson MI order 2)")
    plt.plot(sibloss2ys_order10, sibpubaccs_order10, '-o', markersize=1, label="Public task (Sibson MI order 10)")
    plt.plot(sibloss2ys_order10, sibsecaccs_order10, '-o', markersize=2, label="Private task (Sibson MI order 10)")
    plt.xlabel("Loss on Y(Cross-entropy)")
    plt.ylabel("Task accuracy")
    plt.legend()
    plt.savefig(savedir + "expDy_exp_final_accs_compare.png", bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(KLsecaccs, KLpubaccs, '-o', markersize=2, label="MI")
    plt.plot(sibsecaccs, sibpubaccs, '--', markersize=2, label="Sibson MI order 40")
    plt.plot(sibsecaccs_order2, sibpubaccs_order2, '-d', markersize=2, label="Sibson MI order 2")
    plt.plot(sibsecaccs_order10, sibpubaccs_order10, '-*', markersize=2, label="Sibson MI order 10")
    plt.xlabel("Private Task accuracy")
    plt.ylabel("Public Task accuracy")
    plt.legend()
    plt.savefig(savedir + "expDy_exp_pairs_compare.png", bbox_inches='tight')
    plt.show()
    plt.close()

    #plot_x_scatter_with_theoretic_compare(savedir, 'expDy_exp_final_accs', sibloss2ys, sibpubaccs, sibloss2ys, sibsecaccs, KLloss2ys, KLpubaccs, KLloss2ys, KLsecaccs, 'Final accuracies vs distortion budget on regular Y', 'Loss on Y(Cross-entropy)', 'Task accuracy', 'Regular task(Sibson MI order 20)', 'Private task(Sibson MI order 20)', 'Private task(MI)', 'Regular task(MI)')
    #plot_x_scatter_with_theoretic(savedir, 'expDy_exp_pairs', sibsecaccs, sibpubaccs, KLsecaccs, KLpubaccs, 'Final accuracies private vs regular task', 'Private Task accuracy', 'Regular Task accuracy', 'Sibson MI order 20', 'MI')
    return

def plot_twotask_synth_diff_coefs(datadir='/home/rxiao/data/synthetic/', savedir='twotask_synth_diff_coefs/'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    listdir = os.listdir(datadir)
    # Remove non-directory elements
    if "1d2gaussian.npz" in listdir:
        listdir.remove("1d2gaussian.npz")
    removelist = []
    for elem in listdir:
        if '.gz' in elem:
            removelist.append(elem)
    for elem in removelist:
        listdir.remove(elem)
    
    # Parse hyperparameters
    hyperparams = []
    for dir in listdir:
        if "1D_2_2_twotask_" in dir:
            metric = "sibMI"
            params = dir.split('1D_2_2_twotask_privacy_checkpoints')[1].split('_')
            encode_coef = float(params[0])
            decode_coef = float(params[1])
            order = float(params[2])
            hyperparams.append({'dir': dir,
                                'metric': metric,
                                'encode_coef': encode_coef,
                                'decode_coef': decode_coef,
                                'order': order})

    enc_loss_train=[]
    enc_loss_val=[]
    dec_loss_train=[]
    dec_loss_val=[]
    KL_loss_train=[]
    KL_loss_val=[]
    MI_loss_train=[]
    MI_loss_val=[]
    sibMI_loss_train=[]
    sibMI_loss_val=[]
    pub_dist_train=[]
    pub_dist_val=[]
    sec_dist_train=[]
    sec_dist_val=[]
    pub_acc_train=[]
    pub_acc_val=[]
    sec_acc_train=[]
    sec_acc_val=[]
    x = []
    keys = {}
    order=50
    for params in hyperparams:
        if params['order']==50:
            data = np.load(datadir+params['dir']+'/synth_trainstats.npz')
            keys = data.keys()
            if 'MIloss_train' not in keys:
                print("MI key not in %s"%(params['dir']))
    for params in hyperparams:
        if params['order']==order:
            #pdb.set_trace()
            data = np.load(datadir+params['dir']+'/synth_trainstats.npz')
            keys = data.keys()
            print("Loading %s" % (params['dir']))
            enc_loss_train.extend([data['e_loss_train'][-1]])
            enc_loss_val.extend([data['e_loss_val'][-1]])
            dec_loss_train.extend([data['d_loss_train'][-1]])
            dec_loss_val.extend([data['d_loss_val'][-1]])
            KL_loss_train.extend([data['KLloss_train'][-1]])
            KL_loss_val.extend([data['KLloss_val'][-1]])
            MI_loss_train.extend([data['MIloss_train'][-1]])
            MI_loss_val.extend([data['MIloss_val'][-1]])
            sibMI_loss_train.extend([data['sibMIloss_train'][-1]])
            sibMI_loss_val.extend([data['sibMIloss_val'][-1]])
            pub_dist_train.extend([data['pub_dist_train'][-1]])
            pub_dist_val.extend([data['pub_dist_val'][-1]])
            sec_dist_train.extend([data['sec_dist_train'][-1]])
            sec_dist_val.extend([data['sec_dist_val'][-1]])
            pub_acc_train.extend([data['pub_acc_train'][-1]])
            pub_acc_val.extend([data['pub_acc_val'][-1]])
            sec_acc_train.extend([data['sec_acc_train'][-1]])
            sec_acc_val.extend([data['sec_acc_val'][-1]])
            x.append(params['encode_coef'])

    pdb.set_trace()
    #x = np.array([float(entry.split('privacy_checkpoints')[-1]) for entry in listdir])
    indices = np.argsort(x)
    plot_enc_loss_coef(keys, indices, x, enc_loss_train, enc_loss_val, savedir)
    plot_dec_loss_coef(keys, indices, x, dec_loss_train, dec_loss_val, savedir)
    plot_KL_loss_coef(keys, indices, x, KL_loss_train, KL_loss_val, savedir)
    plot_MI_loss_coef(keys, indices, x, MI_loss_train, MI_loss_val, savedir)
    plot_sibMI_loss_coef(keys, indices, x, sibMI_loss_train, sibMI_loss_val, savedir, order)
    plot_dist_coef(keys, indices, x, pub_dist_train, pub_dist_val, sec_dist_train, sec_dist_val, savedir)
    plot_pub_sec_acc_coef(keys, indices, x, pub_acc_train, pub_acc_val, sec_acc_train, sec_acc_val, savedir)
    return

def plot_synthsib_scatter(datadir='/home/rxiao/data/synthetic/compare_theoretic/', savedir='synth/SibMI_dist/scatter/', dataset='synth'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    listdir = os.listdir(datadir)
    # Remove non-directory elements
    if "1d2gaussian.npz" in listdir:
        listdir.remove("1d2gaussian.npz")
    removelist = []
    for elem in listdir:
        if '.gz' in elem:
            removelist.append(elem)
    for elem in removelist:
        listdir.remove(elem)
    
    # Parse hyperparameters
    # Strip away letters and non-digits
    all = string.maketrans('','')
    nodigits = all.translate(all, string.digits)
    hyperparams = []
    if dataset == 'synth':
        for dir in listdir:
            if "sibMI_privacy" in dir:
                metric = "sibMI"
                params = dir.split('sibMI_privacy_checkpoints')[1].split('_')
                if len(params) == 5:
                    encode_coef = float(params[0])
                    decode_coef = float(params[1])
                    #dist_budget = float(params[2].translate(all, nodigits))
                    dist_budget = float(params[2].strip('D'))
                    order = float(params[3].strip('order'))
                    info_budget = float(params[4].strip('I'))
                    if decode_coef == 1:
                        hyperparams.append({'dir': dir,
                                'metric': metric,
                                'encode_coef': encode_coef,
                                'decode_coef': decode_coef,
                                'order': order,
                                'dist_budget': dist_budget,
                                'info_budget': info_budget})
    if dataset == 'MNIST':
        pdb.set_trace()
        for dir in listdir:
            if "sibMI" in dir and "xmetricPPANMNIST" in dir:
                metric = "sibMI"
                params = dir.split('sibMIprivacy_checkpoints')[1].split('_')
                if len(params) == 5:
                    encode_coef = float(params[0])
                    decode_coef = float(params[1])
                    dist_budget = float(params[2].strip('D'))
                    order = float(params[3].strip('order'))
                    xmetric = (params[4].strip('xmetric'))
                    if decode_coef == 1:
                        hyperparams.append({'dir': dir,
                                'metric': metric,
                                'encode_coef': encode_coef,
                                'decode_coef': decode_coef,
                                'order': order,
                                'dist_budget': dist_budget,
                                'xmetric': xmetric})


    enc_loss_train=[]
    enc_loss_val=[]
    dec_loss_train=[]
    dec_loss_val=[]
    loss2x_train=[]
    loss2x_val=[]
    KL_loss_train=[]
    KL_loss_val=[]
    MI_loss_train=[]
    MI_loss_val=[]
    sibMI_loss_train=[]
    sibMI_loss_val=[]
    pub_dist_train=[]
    pub_dist_val=[]
    sec_dist_train=[]
    sec_dist_val=[]
    sec_acc_train=[]
    sec_acc_val=[]
    dist_budget = []
    info_budget = []
    keys = {}
    order=20
    for params in hyperparams:
        if params['order']==20 and 'synth_trainstats.npz' in os.listdir(datadir+params['dir']):
            data = np.load(datadir+params['dir']+'/synth_trainstats.npz')
            keys = data.keys()
            if 'MIloss_train' not in keys:
                print("MI key not in %s"%(params['dir']))
    for params in hyperparams:
        if params['order']==order and 'synth_trainstats.npz' in os.listdir(datadir+params['dir']):
            #pdb.set_trace()
            data = np.load(datadir+params['dir']+'/synth_trainstats.npz')
            keys = data.keys()
            print("Loading %s" % (params['dir']))
            index = 0
            minacc = 10
            maxacc = 0
            maxindex = 0
            for i in xrange(20, len(data['e_loss_train'])):
                if data['loss2x_train'][i] < params['dist_budget'] and data['sec_acc_train'][i] < minacc:
                    index = i
                    minacc = data['sec_acc_train'][i]
                if data['loss2x_train'][i] < params['dist_budget'] and data['sec_acc_train'][i] > maxacc:
                    maxindex = i
            valindex = int(index - index%10 + 9)
            #pdb.set_trace()
            if index != 0:
                print("Adding sample from run " + params['dir'])
                print("at index %s, L2x train: %s, sibMI train: %s, secacc train: %s" % (valindex, data['loss2x_train'][index], data['sibMIloss_train'][index], data['sec_acc_train'][index]))
                print("at valindex %s, L2x val: %s, sibMI val: %s, secacc val: %s" % (valindex, data['loss2x_val'][valindex], data['sibMIloss_val'][valindex], data['sec_acc_val'][valindex]))
                enc_loss_train.extend([data['e_loss_train'][index]])
                enc_loss_val.extend([data['e_loss_val'][valindex]])
                dec_loss_train.extend([data['d_loss_train'][index]])
                dec_loss_val.extend([data['d_loss_val'][valindex]])
                loss2x_train.extend([data['loss2x_train'][index]])
                loss2x_val.extend([data['loss2x_val'][valindex]])
                KL_loss_train.extend([data['KLloss_train'][index]])
                KL_loss_val.extend([data['KLloss_val'][valindex]])
                MI_loss_train.extend([data['MIloss_train'][index]])
                MI_loss_val.extend([data['MIloss_val'][valindex]])
                sibMI_loss_train.extend([data['sibMIloss_train'][index]])
                sibMI_loss_val.extend([data['sibMIloss_val'][valindex]])
                pub_dist_train.extend([data['pub_dist_train'][index]])
                pub_dist_val.extend([data['pub_dist_val'][valindex]])
                sec_dist_train.extend([data['sec_dist_train'][index]])
                sec_dist_val.extend([data['sec_dist_val'][valindex]])
                sec_acc_train.extend([data['sec_acc_train'][index]])
                sec_acc_val.extend([data['sec_acc_val'][valindex]])
                dist_budget.append(params['dist_budget'])
                if 'info_budget' in keys:
                    info_budget.append(params['info_budget'])

    #x = np.array([float(entry.split('privacy_checkpoints')[-1]) for entry in listdir])
    #indices = np.argsort(x)
    if dataset == 'synth':
        skipindex = (4, 9, 21, 22, 0, 5, 8, 11, 13, 18, 20, 24, 26, 28, 30, 31)
        chooseindex = (2,4,5,9,15,16,18,19,20,23,29,31,32,34,37)
    else:
        skipindex = []
        chooseindex = []
    pdb.set_trace()
    #loss2x_train = [loss2x_train[e] for e in xrange(len(loss2x_train)) if e not in skipindex]
    #sibMI_loss_train = [sibMI_loss_train[e] for e in xrange(len(sibMI_loss_train)) if e not in skipindex]
    #sec_acc_train = [sec_acc_train[e] for e in xrange(len(sec_acc_train)) if e not in skipindex]
    #loss2x_train = [loss2x_train[e] for e in xrange(len(loss2x_train)) if e in chooseindex]
    #sibMI_loss_train = [sibMI_loss_train[e] for e in xrange(len(sibMI_loss_train)) if e in chooseindex]
    #sec_acc_train = [sec_acc_train[e] for e in xrange(len(sec_acc_train)) if e in chooseindex]
    #pdb.set_trace()
    #plot_MI_loss_coef(keys, indices, x, MI_loss_train, MI_loss_val, savedir)
    #plot_sibMI_loss_coef(keys, indices, x, sibMI_loss_train, sibMI_loss_val, savedir, order)
    #plot_dist_coef(keys, indices, x, pub_dist_train, pub_dist_val, sec_dist_train, sec_dist_val, savedir)
    #plot_pub_sec_acc_coef(keys, indices, x, pub_acc_train, pub_acc_val, sec_acc_train, sec_acc_val, savedir)
    #plot_x_scatter(savedir, "dist_vs_sibMI.png", loss2x_train, sibMI_loss_train, legend="Sibson MI vs distortion", xlabel="Experimental Distortion", ylabel="Sibson MI, order=20")
    #plot_x_scatter(savedir, "MI_vs_secacc.png", sibMI_loss_train, sec_acc_train, legend="Private accuracy vs MI", xlabel="Sibson MI, order=20", ylabel="Private accuracy")
    #plot_x_scatter(savedir, "dist_vs_secacc.png", loss2x_train, sec_acc_train, legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy")
    if dataset == 'synth':
        theorydata = np.load("synth/theoretic_upper_bound_alpha5.npz")
    #plot_x_scatter_with_theoretic(savedir, "dist_vs_secacc_with_theoretic.png", loss2x_train, sec_acc_train, theorydata['Ds'], theorydata['phis'], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy")
    PPAN_MNIST_nodiscrim_data = [[0.07, 0.95],
                                 [0.075, 0.93],
                                 [0.08, 0.9],
                                 [0.12, 0.61],
                                 [0.16, 0.38],
                                 [0.17, 0.35],
                                 [0.175, 0.27],
                                 [0.18, 0.24],
                                 [0.19, 0.17],
                                 [0.20, 0.15]]
    PPAN_MNIST_nodiscrim_data = np.array(PPAN_MNIST_nodiscrim_data)
    plot_x_scatter_with_theoretic(savedir, "dist_vs_secacc_with_PPAN.png", loss2x_train, sec_acc_train, PPAN_MNIST_nodiscrim_data[:, 0], PPAN_MNIST_nodiscrim_data[:, 1], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy", label1="Sibson MI", label2="PPAN")
    return

def plot_synth_sibvsKL(sibdatadir='/home/rxiao/data/synthetic/compare_theoretic/sibMI_c_cz/', KLdatadir='/home/rxiao/data/synthetic/compare_theoretic/KL/', savedir='synth/SibMI_vs_KL/scatter/', dataset='synth'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    def extract_params(datadir):
        listdir = os.listdir(datadir)
        # Remove non-directory elements
        if "1d2gaussian.npz" in listdir:
            listdir.remove("1d2gaussian.npz")
        removelist = []
        for elem in listdir:
            if '.gz' in elem:
                removelist.append(elem)
        for elem in removelist:
            listdir.remove(elem)
        
        # Parse hyperparameters
        # Strip away letters and non-digits
        all = string.maketrans('','')
        nodigits = all.translate(all, string.digits)
        hyperparams = []
        if dataset == 'synth':
            for dir in listdir:
                if "sibMI_privacy" in dir:
                    metric = "sibMI"
                    params = dir.split('sibMI_privacy_checkpoints')[1].split('_')
                    if len(params) == 5:
                        encode_coef = float(params[0])
                        decode_coef = float(params[1])
                        #dist_budget = float(params[2].translate(all, nodigits))
                        dist_budget = float(params[2].strip('D'))
                        order = float(params[3].strip('order'))
                        info_budget = float(params[4].strip('I'))
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    'dist_budget': dist_budget,
                                    'info_budget': info_budget})
                if "KL_privacy" in dir:
                    metric = "KL"
                    params = dir.split('KL_privacy_checkpoints')[1].split('_')
                    if len(params) == 5:
                        encode_coef = float(params[0])
                        decode_coef = float(params[1])
                        #dist_budget = float(params[2].translate(all, nodigits))
                        dist_budget = float(params[2].strip('D'))
                        order = float(params[3].strip('order'))
                        info_budget = float(params[4].strip('I'))
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    'dist_budget': dist_budget,
                                    'info_budget': info_budget})

        if dataset == 'MNIST':
            #pdb.set_trace()
            for dir in listdir:
                if "sibMI" in dir and "xmetricL2" in dir:
                    metric = "sibMI"
                    params = dir.split('sibMIprivacy_checkpoints')[1].split('_')
                    if len(params) == 5:
                        encode_coef = float(params[0])
                        decode_coef = float(params[1])
                        dist_budget = float(params[2].strip('D'))
                        order = float(params[3].strip('order'))
                        xmetric = (params[4].strip('xmetric'))
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    'dist_budget': dist_budget,
                                    'xmetric': xmetric})
                if "KLprivacy" in dir:
                    metric = "KL"
                    params = dir.split('KLprivacy_checkpoints')[1].split('_')
                    if len(params) == 5:
                        encode_coef = float(params[0])
                        decode_coef = float(params[1])
                        #dist_budget = float(params[2].translate(all, nodigits))
                        dist_budget = float(params[2].strip('D'))
                        order = float(params[3].strip('order'))
                        #info_budget = float(params[4].strip('I'))
                        xmetric = (params[4].strip('xmetric'))
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    'dist_budget': dist_budget,
                                    #'info_budget': info_budget,
                                    'xmetric': xmetric})



        enc_loss_train=[]
        enc_loss_val=[]
        dec_loss_train=[]
        dec_loss_val=[]
        loss2x_train=[]
        loss2x_val=[]
        KL_loss_train=[]
        KL_loss_val=[]
        MI_loss_train=[]
        MI_loss_val=[]
        sibMI_loss_train=[]
        sibMI_loss_val=[]
        pub_dist_train=[]
        pub_dist_val=[]
        sec_dist_train=[]
        sec_dist_val=[]
        sec_acc_train=[]
        sec_acc_val=[]
        dist_budget = []
        info_budget = []
        keys = {}
        order=20
        for params in hyperparams:
            if params['order']==20 and 'synth_trainstats.npz' in os.listdir(datadir+params['dir']):
                data = np.load(datadir+params['dir']+'/synth_trainstats.npz')
                keys = data.keys()
                if 'MIloss_train' not in keys:
                    print("MI key not in %s"%(params['dir']))
        for params in hyperparams:
            if params['order']==order and 'synth_trainstats.npz' in os.listdir(datadir+params['dir']):
                data = np.load(datadir+params['dir']+'/synth_trainstats.npz')
                keys = data.keys()
                print("Loading %s" % (params['dir']))
                index = 0
                minacc = 10
                maxacc = 0
                maxindex = 0
                for i in xrange(len(data['loss2x_train'])-1, 20, -1):
                    if data['loss2x_train'][i] < params['dist_budget'] and data['sec_acc_train'][i] < minacc:
                        index = i
                        minacc = data['sec_acc_train'][i]
                    if data['loss2x_train'][i] < params['dist_budget'] and data['sec_acc_train'][i] > maxacc:
                        maxindex = i
                #if params['dist_budget'] == 0.04:
                #    pdb.set_trace()
                #index = len(data['loss2x_train']) - 1
                if params['metric'] == 'sibMI': 
                    index = 40
                if params['metric'] == 'KL': 
                    index = len(data['loss2x_train']) - 1
                valindex = int(index - index%10 + 9)
                if index != 0:
                    print("Adding sample from run " + params['dir'])
                    print("at index %s, L2x train: %s, sibMI train: %s, secacc train: %s" % (valindex, data['loss2x_train'][index], data['sibMIloss_train'][index], data['sec_acc_train'][index]))
                    print("at valindex %s, L2x val: %s, sibMI val: %s, secacc val: %s" % (valindex, data['loss2x_val'][valindex], data['sibMIloss_val'][valindex], data['sec_acc_val'][valindex]))
                    enc_loss_train.extend([data['e_loss_train'][index]])
                    enc_loss_val.extend([data['e_loss_val'][valindex]])
                    dec_loss_train.extend([data['d_loss_train'][index]])
                    dec_loss_val.extend([data['d_loss_val'][valindex]])
                    loss2x_train.extend([data['loss2x_train'][index]])
                    loss2x_val.extend([data['loss2x_val'][valindex]])
                    KL_loss_train.extend([data['KLloss_train'][index]])
                    KL_loss_val.extend([data['KLloss_val'][valindex]])
                    MI_loss_train.extend([data['MIloss_train'][index]])
                    MI_loss_val.extend([data['MIloss_val'][valindex]])
                    sibMI_loss_train.extend([data['sibMIloss_train'][index]])
                    sibMI_loss_val.extend([data['sibMIloss_val'][valindex]])
                    pub_dist_train.extend([data['pub_dist_train'][index]])
                    pub_dist_val.extend([data['pub_dist_val'][valindex]])
                    sec_dist_train.extend([data['sec_dist_train'][index]])
                    sec_dist_val.extend([data['sec_dist_val'][valindex]])
                    sec_acc_train.extend([data['sec_acc_train'][index]])
                    sec_acc_val.extend([data['sec_acc_val'][valindex]])
                    dist_budget.append(params['dist_budget'])
                    if 'info_budget' in keys:
                        info_budget.append(params['info_budget'])
        return {"enc_loss_train": enc_loss_train,
                "enc_loss_val": enc_loss_val,
                "dec_loss_train": dec_loss_train,
                "dec_loss_val": dec_loss_val,
                "loss2x_train": loss2x_train,
                "loss2x_val": loss2x_val,
                "KL_loss_train": KL_loss_train,
                "KL_loss_val": KL_loss_val,
                #MI_loss_train=[]
                #MI_loss_val=[]
                "sibMI_loss_train": sibMI_loss_train,
                "sibMI_loss_val": sibMI_loss_val,
                #pub_dist_train=[]
                #pub_dist_val=[]
                #sec_dist_train=[]
                #sec_dist_val=[]
                "sec_acc_train": sec_acc_train,
                "sec_acc_val": sec_acc_val,
                "dist_budget": dist_budget}
                #info_budget = []
    sibdata = extract_params(sibdatadir)
    KLdata = extract_params(KLdatadir)

    sibdata_order2 = [[0.02, 0.8935, 0.01938],
                      [0.03, 0.7058, 0.02924],
                      [0.05, 0.4420, 0.04880],
                      [0.06, 0.3104, 0.05863],
                      [0.07, 0.1509, 0.06748]]
    sibdata_order10 = [[0.01, 0.9632, 0.01230],
                       [0.02, 0.8954, 0.01986],
                       [0.03, 0.6925, 0.02642],
                       [0.04, 0.6037, 0.03884],
                       [0.05, 0.4457, 0.04861],
                       [0.06, 0.2840, 0.05861],
                       [0.07, 0.1260, 0.06740]]
    if dataset == 'MNIST':
        pdb.set_trace()
        indices = np.argsort(sibdata["dist_budget"])[:-2]
        distbudget = np.array(sibdata["dist_budget"])[indices]
        secacc = np.array(sibdata["sec_acc_train"])[indices]
        plt.plot(distbudget, secacc, '-o', label="Sibson MI, order 20")
        indices = np.argsort(KLdata["dist_budget"])
        distbudget = np.array(KLdata["dist_budget"])[indices]
        secacc = np.array(KLdata["sec_acc_train"])[indices]
        plt.plot(distbudget, secacc, '--', label="MI")
        plt.plot(np.array(sibdata_order2)[:, 0], np.array(sibdata_order2)[:, 1], '-d', label="Sibson MI, order 2")
        plt.plot(np.array(sibdata_order10)[:, 0], np.array(sibdata_order10)[:, 1], label="Sibson MI, order 10")
        plt.xlabel("Distortion")
        plt.ylabel("Private accuracy")
        plt.legend()
        plt.savefig(savedir+'/compare_priv_acc_multiorder.png', bbox_inches='tight')
        plt.show()
        plt.close()
 
    #plot_x_scatter_with_theoretic(savedir, "dist_vs_secacc_KL_sibMI.png", sibdata["loss2x_train"], sibdata["sec_acc_train"], KLdata["loss2x_train"], KLdata["sec_acc_train"], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy", label1="SibMI", label2="KL")
    plot_x_scatter_with_theoretic(savedir, "distbudget_vs_secacc_KL_sibMI.png", sibdata["dist_budget"], sibdata["sec_acc_train"], KLdata["dist_budget"], KLdata["sec_acc_train"], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy", label1="SibMI", label2="MI")
    return


def plot_ferg_sibvsKL(sibdatadir='/home/rxiao/data/ferg/remote_exp/ferg/sibMIgen/', KLdatadir='/home/rxiao/data/ferg/remote_exp/ferg/KLgen/', savedir='ferg/SibMI_vs_KL/scatter/', dataset='ferg', flag='None'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    def extract_params(datadir):
        listdir = os.listdir(datadir)
        # Remove non-directory elements
        if "1d2gaussian.npz" in listdir:
            listdir.remove("1d2gaussian.npz")
        removelist = []
        for elem in listdir:
            if '.gz' in elem:
                removelist.append(elem)
        for elem in removelist:
            listdir.remove(elem)
        
        # Parse hyperparameters
        # Strip away letters and non-digits
        all = string.maketrans('','')
        nodigits = all.translate(all, string.digits)
        hyperparams = []
        if dataset == 'ferg':
            #pdb.set_trace()
            for dir in listdir:
                if "sibMI" in dir and dir.endswith("_40"):
                    metric = "sibMI"
                    params = dir.split('privacy_checkpoints')[1].split('_')[1:]
                    if len(params) == 7:
                        encode_coef = float(params[4])
                        decode_coef = float(params[5])
                        distx_budget = float(params[1]) if params[0] == "Dx" else 0.0
                        disty_budget = float(params[3]) if params[2] == "Dy" else 0.0
                        order = float(params[6])
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    'distx_budget': distx_budget,
                                    'disty_budget': disty_budget})
                    if len(params) == 5:
                        disty_budget = float(params[1]) if params[0] == "Dy" else 0.0
                        encode_coef = float(params[2])
                        decode_coef = float(params[3])
                        order = float(params[4])
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    'disty_budget': disty_budget})

                if "KL" in dir:
                    metric = "KL"
                    params = dir.split('privacy_checkpoints')[1].split('_')[1:]
                    if len(params) == 7:
                        encode_coef = float(params[4])
                        decode_coef = float(params[5])
                        #dist_budget = float(params[2].translate(all, nodigits))
                        distx_budget = float(params[1]) if params[0] == "Dx" else 0.0
                        disty_budget = float(params[3]) if params[2] == "Dy" else 0.0
                        order = float(params[6])
                        #info_budget = float(params[4].strip('I'))
                        #xmetric = (params[4].strip('xmetric'))
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    #'info_budget': info_budget,
                                    'distx_budget': distx_budget,
                                    'disty_budget': disty_budget})
                    if len(params) == 5:
                        disty_budget = float(params[1]) if params[0] == "Dy" else 0.0
                        encode_coef = float(params[2])
                        decode_coef = float(params[3])
                        order = float(params[4])
                        if decode_coef == 1:
                            hyperparams.append({'dir': dir,
                                    'metric': metric,
                                    'encode_coef': encode_coef,
                                    'decode_coef': decode_coef,
                                    'order': order,
                                    'disty_budget': disty_budget})

        enc_loss_train=[]
        enc_loss_val=[]
        dec_loss_train=[]
        dec_loss_val=[]
        loss2x_train=[]
        loss2x_val=[]
        KL_loss_train=[]
        KL_loss_val=[]
        MI_loss_train=[]
        MI_loss_val=[]
        sibMI_loss_train=[]
        sibMI_loss_val=[]
        pub_dist_train=[]
        pub_dist_val=[]
        sec_dist_train=[]
        sec_dist_val=[]
        pub_acc_train=[]
        pub_acc_val=[]
        sec_acc_train=[]
        sec_acc_val=[]
        dist_budget = []
        info_budget = []
        keys = {}
        order=40
        for params in hyperparams:
            if params['order']==40 and 'ferg_trainstats.npz' in os.listdir(datadir+params['dir']):
                data = np.load(datadir+params['dir']+'/ferg_trainstats.npz')
                keys = data.keys()
                if 'MIloss_train' not in keys:
                    print("MI key not in %s"%(params['dir']))
        for params in hyperparams:
            if params['order']==order and 'ferg_trainstats.npz' in os.listdir(datadir+params['dir']):
                data = np.load(datadir+params['dir']+'/ferg_trainstats.npz')
                keys = data.keys()
                print("Loading %s" % (params['dir']))
                index = 0
                minacc = 10
                maxacc = 0
                maxindex = 0
                if params['metric'] == 'sibMI' and params['distx_budget'] <= 0.02: 
                    #pdb.set_trace()
                    for i in xrange(len(data['loss2x_train'])-1, 10, -1):
                        if 'distx_budget' in params.keys():
                            if data['loss2x_train'][i] < params['distx_budget'] and data['sec_acc_train'][i] < minacc:
                                index = i
                                minacc = data['sec_acc_train'][i]
                        if 'distx_budget' not in params.keys() and data['loss2y_train'][i] < params['disty_budget'] and data['sec_acc_train'][i] < minacc:
                            index = i
                            minacc = data['sec_acc_train'][i]
                #if params['dist_budget'] == 0.04:
                #    pdb.set_trace()
                #index = len(data['loss2x_train']) - 1
                #pdb.set_trace()
                #if params['metric'] == 'sibMI': 
                #    index = 40
                if params['metric'] == 'KL':
                    #pdb.set_trace()
                    #for i in xrange(len(data['loss2x_train'])-1, 10, -1):
                        #if data['loss2x_train'][i] < params['distx_budget']:
                        #    index = i
                        #    minacc = data['sec_acc_train'][i]
                        #if data['loss2x_train'][i] < params['distx_budget'] and data['sec_acc_train'][i] > maxacc:
                        #    index = i
                        #    maxindex = i
                    #index = len(data['loss2x_train']) - 55
                    index = 100
                valindex = int(index - index%10 + 9)
                if index != 0:
                    print("Adding sample from run " + params['dir'])
                    print("at index %s, L2x train: %s, sibMI train: %s, secacc train: %s" % (valindex, data['loss2x_train'][index], data['sibMIloss_train'][index], data['sec_acc_train'][index]))
                    print("at valindex %s, L2x val: %s, sibMI val: %s, secacc val: %s" % (valindex, data['loss2x_val'][valindex], data['sibMIloss_val'][valindex], data['sec_acc_val'][valindex]))
                    enc_loss_train.extend([data['e_loss_train'][index]])
                    enc_loss_val.extend([data['e_loss_val'][valindex]])
                    #dec_loss_train.extend([data['g_loss_train'][index]])
                    #dec_loss_val.extend([data['g_loss_val'][valindex]])
                    loss2x_train.extend([data['loss2x_train'][index]])
                    loss2x_val.extend([data['loss2x_val'][valindex]])
                    KL_loss_train.extend([data['KLloss_train'][index]])
                    KL_loss_val.extend([data['KLloss_val'][valindex]])
                    MI_loss_train.extend([data['MIloss_train'][index]])
                    MI_loss_val.extend([data['MIloss_val'][valindex]])
                    sibMI_loss_train.extend([data['sibMIloss_train'][index]])
                    sibMI_loss_val.extend([data['sibMIloss_val'][valindex]])
                    pub_dist_train.extend([data['pub_dist_train'][index]])
                    pub_dist_val.extend([data['pub_dist_val'][valindex]])
                    sec_dist_train.extend([data['sec_dist_train'][index]])
                    sec_dist_val.extend([data['sec_dist_val'][valindex]])
                    pub_acc_train.extend([data['pub_acc_train'][index]])
                    pub_acc_val.extend([data['pub_acc_val'][valindex]])
                    sec_acc_train.extend([data['sec_acc_train'][index]])
                    sec_acc_val.extend([data['sec_acc_val'][valindex]])
                    #if 'distx_budget' in keys and 'disty_budget' not in keys:
                    if 'distx_budget' in params.keys():# and 'disty_budget' not in keys:
                        dist_budget.append(params['distx_budget'])
                    if 'distx_budget' not in keys and 'disty_budget' in keys:
                        dist_budget.append(params['disty_budget'])
                    if 'info_budget' in keys:
                        info_budget.append(params['info_budget'])
        return {"enc_loss_train": enc_loss_train,
                "enc_loss_val": enc_loss_val,
                "dec_loss_train": dec_loss_train,
                "dec_loss_val": dec_loss_val,
                "loss2x_train": loss2x_train,
                "loss2x_val": loss2x_val,
                "KL_loss_train": KL_loss_train,
                "KL_loss_val": KL_loss_val,
                #MI_loss_train=[]
                #MI_loss_val=[]
                "sibMI_loss_train": sibMI_loss_train,
                "sibMI_loss_val": sibMI_loss_val,
                #pub_dist_train=[]
                #pub_dist_val=[]
                #sec_dist_train=[]
                #sec_dist_val=[]
                "pub_acc_train": pub_acc_train,
                "pub_acc_val": pub_acc_val,
                "sec_acc_train": sec_acc_train,
                "sec_acc_val": sec_acc_val,
                "dist_budget": dist_budget}
                #info_budget = []
    pdb.set_trace()
    if os.path.isfile(os.path.join(savedir, "ferg_sib_vs_KL_sibdata.pickle")):
        with open(os.path.join(savedir, "ferg_sib_vs_KL_sibdata.pickle"), 'rb') as handle:
            sibdata = pickle.load(handle)
        sibdata["dist_budget"] = np.array([sibdata["dist_budget"][i] for i in [0, 1, 3, 5, 6, 7, 8, 11, 12]])
        sibdata["sec_acc_val"] = np.array([sibdata["sec_acc_val"][i] for i in [0, 1, 3, 5, 6, 7, 8, 11, 12]])
    if os.path.isfile(os.path.join(savedir, "ferg_sib_vs_KL_KLdata.pickle")):
        with open(os.path.join(savedir, "ferg_sib_vs_KL_KLdata.pickle"), 'rb') as handle:
            KLdata = pickle.load(handle)
    else:
        sibdata = extract_params(sibdatadir)
        KLdata = extract_params(KLdatadir)
    #plot_x_scatter_with_theoretic(savedir, "dist_vs_secacc_KL_sibMI.png", sibdata["loss2x_train"], sibdata["sec_acc_train"], KLdata["loss2x_train"], KLdata["sec_acc_train"], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy", label1="SibMI", label2="KL")
    #plot_x_scatter_with_theoretic(savedir, "distbudget_vs_secacc_KL_sibMI.png", sibdata["dist_budget"], sibdata["sec_acc_train"], KLdata["dist_budget"], KLdata["sec_acc_train"], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy", label1="SibMI", label2="KL")
    #plot_x_scatter_with_theoretic(savedir, "distbudget_vs_secacc_KL_sibMI_val.png", sibdata["dist_budget"], sibdata["sec_acc_val"], KLdata["dist_budget"], KLdata["sec_acc_val"], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy", label1="SibMI", label2="KL")
    # plot for regular and private tasks on sibMI and KL
    plot_x_scatter_with_theoretic(savedir, "distbudget_vs_pubacc_KL_sibMI_val.png", sibdata["dist_budget"], sibdata["pub_acc_val"], KLdata["dist_budget"], KLdata["pub_acc_val"], legend="Distortion budget vs Public accuracy", xlabel="Distortion", ylabel="Public accuracy", label1="SibMI", label2="MI")
    plot_x_scatter_with_theoretic(savedir, "distbudget_vs_secacc_KL_sibMI_val.png", sibdata["dist_budget"], sibdata["sec_acc_val"], KLdata["dist_budget"], KLdata["sec_acc_val"], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy", label1="SibMI", label2="MI")
    #plot_x_scatter(savedir, "Secacc_vs_distx", sibdata['dist_budget'], sibdata['sec_acc_train'], legend="Distortion budget vs Private accuracy", xlabel="Distortion", ylabel="Private accuracy")
    #plot_x_scatter(savedir, "SibMI_vs_distx", sibdata['dist_budget'], sibdata['sibMI_loss_val'], legend="Distortion budget vs Sibson MI", xlabel="Distortion", ylabel="Sibson MI")
    #np.savez(os.path.join(savedir, "ferg_sib_vs_KL_data"), sibdata=sibdata, KLdata=KLdata)
    # try to save as a pickle
    with open(os.path.join(savedir, "ferg_sib_vs_KL_sibdata.pickle"), 'wb') as handle:
        pickle.dump(sibdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(savedir, "ferg_sib_vs_KL_KLdata.pickle"), 'wb') as handle:
        pickle.dump(KLdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def plot_sibson_synth(data, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #pdb.set_trace()
    # extract the first row of the probability vectors/matrices
    keys = data.keys()
    if 'sibMI_pz' in keys:
        plt.plot(data['pz'][:,0], data['sibMI_pz'], label='SibsonMI')
        plt.legend()
        plt.title('Sibson MI vs binary P(Z) distribution (p, 1-p)')
        plt.xlabel('p')
        plt.ylabel('Sibson MI')
        plt.savefig(savedir + 'sibsonMI_pz.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'sibMI_pcz' in keys:
        plt.plot(data['pcz'][:,0], data['sibMI_pcz'], label='SibsonMI')
        plt.legend()
        plt.title('Sibson MI vs binary P(C|Z) distribution (p, (1-p)/3, (1-p)/3, (1-p)/3)')
        plt.xlabel('p')
        plt.ylabel('Sibson MI')
        plt.savefig(savedir + 'sibsonMI_pcz.png', bbox_inches='tight')
        plt.show()
        plt.close()

    if 'sibMI_alpha' in keys and 'KL_alpha' in keys:
        plt.plot(data['alpha'], data['sibMI_alpha'], '--', markersize=2, label='SibsonMI')
        plt.plot(data['alpha'], data['KL_alpha'], '-d', markersize=2, label='KL')
        plt.legend()
        plt.title('Sibson MI vs alpha')
        plt.xlabel('alpha')
        plt.ylabel('Sibson MI')
        plt.savefig(savedir + 'sibsonMI_alpha.png', bbox_inches='tight')
        plt.show()
        plt.close()
    return

def plot_sibson_comp(data, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #pdb.set_trace()
    # extract the first row of the probability vectors/matrices
    plt.plot(data['pz'][:,0], data['sumz_out_pz'], label='Sum out')
    plt.plot(data['pz'][:,0], data['sumz_in_pz'], label='Sum in')
    plt.legend()
    plt.title('Comparison of sum out vs sum in against binary P(Z) distribution (p, 1-p)')
    plt.xlabel('p')
    plt.ylabel('Sum')
    plt.savefig(savedir + 'sum_pz.png', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(data['pcz'][:,0], data['sumz_out_pcz'], label='Sum out')
    plt.plot(data['pcz'][:,0], data['sumz_in_pcz'], label='Sum in')
    plt.legend()
    plt.title('Comparison of sum out vs sum in against binary P(C|Z) distribution (p, (1-p)/3, (1-p)/3, (1-p)/3)')
    plt.xlabel('p')
    plt.ylabel('Sum')
    plt.savefig(savedir + 'sum_pcz.png', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(data['alpha'], data['sumz_out_alpha'], label='SibsonMI')
    plt.plot(data['alpha'], data['sumz_in_alpha'], label='SibsonMI')
    plt.legend()
    plt.title('Comparison of sum out vs sum in against alpha')
    plt.xlabel('alpha')
    plt.ylabel('Sum')
    plt.savefig(savedir + 'sum_alpha.png', bbox_inches='tight')
    plt.show()
    plt.close()
    return




def main():
    #plot_celeba_perf()
    #plot_synth_perf()
    # MNIST dataset plots
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/privacy_checkpoints1e-08/synth_trainstats.npz'))
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/privacy_checkpoints100/synth_trainstats.npz'), savedir='mnist/encode_100/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/privacy_checkpoints1e-06/synth_trainstats.npz'), savedir='mnist/encode_1e-06/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/privacy_checkpoints1/synth_trainstats.npz'), savedir='mnist/encode_1/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/privacy_checkpoints10/synth_trainstats.npz'), savedir='mnist/encode_10/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/KLprivacy_checkpoints70/synth_trainstats.npz'), savedir='mnist/encode_70/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/KLprivacy_checkpoints40/synth_trainstats.npz'), savedir='mnist/encode_40/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/KLprivacy_checkpoints30/synth_trainstats.npz'), savedir='mnist/encode_30/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/sibMIprivacy_checkpoints100_1_1.01/synth_trainstats.npz'), savedir='mnist/order1.01/encode_100/', order=1.01)
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/sibMIprivacy_checkpoints100_1_10/synth_trainstats.npz'), savedir='mnist/order10/encode_100/', order=10)
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/sibMIprivacy_checkpoints100_1_3/synth_trainstats.npz'), savedir='mnist/order3/encode_100/', order=3)
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/sibMIprivacy_checkpoints100_1_50/synth_trainstats.npz'), savedir='mnist/order50/encode_100/', order=50)
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/sibMIprivacy_checkpoints100_100_20/synth_trainstats.npz'), savedir='mnist/order20/encode_100/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/sibMIprivacy_checkpoints100_100_20/synth_trainstats.npz'), savedir='mnist/order20/encode_100_decode_100/', order=20)
    # MNIST experiments with sibMI and distortion from PPAN
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/sibMIprivacy_checkpoints1_1_D0.3_order20/synth_trainstats.npz'), savedir='mnist/SibMI_dist/1_1_D0.3_order20/', order=20)
    # MNIST remote experiments with sibMI and PPAN distortion, for D=0.1, 0.15, 0.2, 0.5
    #plot_synth_perf(data=np.load('/home/rxiao/data/mnist/remote_exp/sibMIprivacy_checkpoints1_1_D0.5_order20_xmetricPPANMNIST/synth_trainstats.npz'), savedir='mnist/SibMI_dist/1_1_D0.5_order20_xmetricPPANMNIST/', order=20)
    # MNIST remote experiments comparing sibMI and KL for distortions from 0.02 to 0.1, using average L2 as metric
    #plot_synth_sibvsKL(sibdatadir='/home/rxiao/data/mnist/remote_exp/', KLdatadir='/home/rxiao/data/mnist/remote_exp/KL/', savedir='mnist/SibMI_vs_KL/L2_scatter/', dataset='MNIST')
    #plot_synth_sibvsKL(sibdatadir='/home/rxiao/data/mnist/remote_exp/sibMI_c_cz/', KLdatadir='/home/rxiao/data/mnist/remote_exp/KL/', savedir='mnist/SibMI_vs_KL/L2_scatter/', dataset='MNIST')
    # MNIST experiment visualizations
    #mnistvisdata=np.load('/home/rxiao/data/mnist/remote_exp/sibMIprivacy_checkpoints1_1_D0.06_order20_xmetricL2/vis_x_xhat.npz')
    #n,dims = mnistvisdata['x'].shape
    #rows, cols = (3,2)
    ##indices = np.random.randint(0, n, size=rows*cols)  # generate random indices for images/recons
    #indices = np.arange(rows*cols)
    #pdb.set_trace()
    #plot_ferg_x_xhat(mnistvisdata, 
    #               indices,
    #               savedir='mnist/visuals/sibMIprivacy_checkpoints1_1_D0.06_order20_xmetricL2/', rows=rows, cols=cols)
    #mnistvisdata=np.load('/home/rxiao/data/mnist/remote_exp/sibMIprivacy_checkpoints1_1_D0.08_order20_xmetricL2/vis_x_xhat.npz')
    #plot_ferg_x_xhat(mnistvisdata, 
    #               indices,
    #               savedir='mnist/visuals/sibMIprivacy_checkpoints1_1_D0.08_order20_xmetricL2/', rows=rows, cols=cols)
    #mnistvisdata=np.load('/home/rxiao/data/mnist/remote_exp/sibMIprivacy_checkpoints1_1_D0.04_order20_xmetricL2/vis_x_xhat.npz')
    #plot_ferg_x_xhat(mnistvisdata, 
    #               indices,
    #               savedir='mnist/visuals/sibMIprivacy_checkpoints1_1_D0.04_order20_xmetricL2/', rows=rows, cols=cols)
 
    # FERG dataset plots
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/KLprivacy_checkpoints1/ferg_trainstats.npz'), savedir='ferg/encode_1/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/KLprivacy_checkpoints5/ferg_trainstats.npz'), savedir='ferg/encode_5/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/KLprivacy_checkpoints20/ferg_trainstats.npz'), savedir='ferg/encode_20/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/KLprivacy_checkpoints30/ferg_trainstats.npz'), savedir='ferg/encode_30/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/KLprivacy_checkpoints40/ferg_trainstats.npz'), savedir='ferg/encode_40/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/KLprivacy_checkpoints50/ferg_trainstats.npz'), savedir='ferg/encode_50/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/sibMIgen_privacy_checkpoints10_1_20/ferg_trainstats.npz'), savedir='ferg/order20/encode_10/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/remote_exp/ferg/KLgen/KLgen_privacy_checkpoints_Dx_0.012_Dy_0.1_1_1_40/ferg_trainstats.npz'), savedir='ferg/KLgen/KLgen_privacy_checkpoints_Dx_0.012_Dy_0.1_1_1_40/', order=40)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/remote_exp/ferg/sibMIprivacy_checkpoints_Dx_0.015_Dy_0.1_1_1_40/ferg_trainstats.npz'), savedir='ferg/KLgen/KLgen_privacy_checkpoints_Dx_0.015_Dy_0.1_1_1_40/', order=40)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/remote_exp/ferg/sibMI/sibMIprivacy_checkpoints_Dy_0.74_1_1_40/ferg_trainstats.npz'), savedir='ferg/sibMIprivacy_checkpoints_Dy_0.74_1_1_40/', order=40)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/remote_exp/ferg/KL/KLprivacy_checkpoints_Dy_0.66_1_1_40/ferg_trainstats.npz'), savedir='ferg/KLprivacy_checkpoints_Dy_0.66_1_1_40/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/remote_exp/ferg/KL/KLprivacy_checkpoints_Dy_0.7_1_1_40/ferg_trainstats.npz'), savedir='ferg/KLprivacy_checkpoints_Dy_0.7_1_1_40/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/remote_exp/ferg/KL/KLprivacy_checkpoints_Dy_0.72_1_1_40/ferg_trainstats.npz'), savedir='ferg/KLprivacy_checkpoints_Dy_0.72_1_1_40/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/remote_exp/ferg/KL/KLprivacy_checkpoints_Dy_0.74_1_1_40/ferg_trainstats.npz'), savedir='ferg/KLprivacy_checkpoints_Dy_0.74_1_1_40/', order=20)
    #plot_ferg_diff_coefs()
    # FERG remote experiments comparing sibMI and KL for distortions from 0.01 to 0.02, using average L2 as metric
    #plot_ferg_sibvsKL()
    # FERG experiments for range of lossy between 0.6 and 0.74
    #plot_ferg_sibvsKL(sibdatadir='/home/rxiao/data/ferg/remote_exp/ferg/sibMIgen/', KLdatadir='/home/rxiao/data/ferg/remote_exp/ferg/KLgen/', savedir='ferg/SibMI_vs_KL/scatter/', dataset='ferg')
    #plot_ferg_Dy_experiments()
    report_ferg_dy_experiments()
    # FERG experiment visualizations
    #fergvisdata=np.load('/home/rxiao/data/ferg/remote_exp/ferg/sibMIgen/sibMIgen_privacy_checkpoints_Dx_0.006_Dy_0.1_1_1_40/vis_x_xhat.npz')
    #n,dims = fergvisdata['x'].shape
    #rows, cols = (3,2)
    #indices = np.random.randint(0, n, size=rows*cols)  # generate random indices for images/recons
    #plot_ferg_x_xhat(fergvisdata, 
    #               indices,
    #               savedir='ferg/visuals/SibMIgen/sibMIgen_privacy_checkpoints_Dx_0.006_Dy_0.1_1_1_40/', rows=rows, cols=cols)
    #fergvisdata=np.load('/home/rxiao/data/ferg/remote_exp/ferg/sibMIgen/sibMIgen_privacy_checkpoints_Dx_0.012_Dy_0.6_1_1_40/vis_x_xhat.npz')
    #plot_ferg_x_xhat(fergvisdata, 
    #               indices,
    #               savedir='ferg/visuals/SibMIgen/sibMIgen_privacy_checkpoints_Dx_0.012_Dy_0.6_1_1_40/', rows=rows, cols=cols)
    #fergvisdata=np.load('/home/rxiao/data/ferg/remote_exp/ferg/sibMIgen/sibMIgen_privacy_checkpoints_Dx_0.01_Dy_0.6_1_1_40/vis_x_xhat.npz')
    #plot_ferg_x_xhat(fergvisdata, 
    #               indices,
    #               savedir='ferg/visuals/SibMIgen/sibMIgen_privacy_checkpoints_Dx_0.01_Dy_0.6_1_1_40/', rows=rows, cols=cols)
    
    # Synthetic data plots
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic_weighted/discrim_privacy_checkpoints10_1/synth_trainstats.npz'), savedir='synthetic_weighted/encode_10/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic_weighted/discrim_privacy_checkpoints100_0.01/synth_trainstats.npz'), savedir='synthetic_weighted/encode_100_0.01/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic_weighted/discrim_privacy_checkpoints1000_0.001/synth_trainstats.npz'), savedir='synthetic_weighted/encode_1000_0.001/')
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic_weighted/weighted_privacy_checkpoints1000_0.001/synth_trainstats.npz'), savedir='synthetic_weighted/nodiscrim_encode_1000_0.001/')
    #plot_mnist_diff_coefs()
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic_weighted/discrim_MI_privacy_checkpoints50_0.001/synth_trainstats.npz'), savedir='synthetic_weighted/discrim_encode_50_0.001/')

    # Sibson MI synth data experiments
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints1_1_1.001/synth_trainstats.npz'), savedir='synthetic/Sibson_order1.001/sibMIencode_1_1/', order=1.001)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints0.001_1_1.001/synth_trainstats.npz'), savedir='synthetic/Sibson_order1.001/sibMIencode_0.001_1/', order=1.001)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints0.001_1_2/synth_trainstats.npz'), savedir='synthetic/Sibson_order2/sibMIencode_0.001_1/', order=2)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints0.001_1_1.1/synth_trainstats.npz'), savedir='synthetic/Sibson_order1.1/sibMIencode_0.001_1/', order=1.1)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints0.001_1_20/synth_trainstats.npz'), savedir='synthetic/Sibson_order20/sibMIencode_0.001_1/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints0.1_1_20/synth_trainstats.npz'), savedir='synthetic/Sibson_order20/sibMIencode_0.1_1/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints1_1_20/synth_trainstats.npz'), savedir='synthetic/Sibson_order20/sibMIencode_1_1/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints10_1_20/synth_trainstats.npz'), savedir='synthetic/Sibson_order20/sibMIencode_10_1/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints0.001_1_10/synth_trainstats.npz'), savedir='synthetic/Sibson_order10/sibMIencode_0.001_1/', order=10)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints0.001_1_1.01/synth_trainstats.npz'), savedir='synthetic/Sibson_order1.01/sibMIencode_0.001_1/', order=1.01)
    # Synthetic two task Gaussian data experiments
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/1D_2_2_twotask_privacy_checkpoints0.1_1_50/synth_trainstats.npz'), savedir='synthetic/1D_twotask/Sibson_order50/sibMIencode_0.1_1/', order=50)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/sibMIgen_privacy_checkpoints100_1_20/synth_trainstats.npz'), savedir='ferg/order20/encode_100/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/ferg/sibMIgen_privacy_checkpoints100_100_20/synth_trainstats.npz'), savedir='ferg/order20/encode_100_decode_100/', order=20)
    #plot_synthsib_diff_coefs()
    #plot_twotask_synth_diff_coefs()
    
    # CGAP experiments
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_CGAP_privacy_checkpoints100_0.1_20/synth_trainstats.npz'), savedir='synth/CGAP/Kiter_100_D_0.1/', order=20)
    # SibMI experiments with same distortion constraints as CGAP
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints20_1_D0.3_order20/synth_trainstats.npz'), savedir='synth/SibMI_dist/20_1_D0.3_order20/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints20_1_D0.05_order20/synth_trainstats.npz'), savedir='synth/SibMI_dist/20_1_D0.05_order20/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints20_1_D0.1_order20/synth_trainstats.npz'), savedir='synth/SibMI_dist/20_1_D0.1_order20/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/sibMI_privacy_checkpoints1_1_D0.001_order20/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D0.001_order20/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D5_order20/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D5_order20/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D1_order20/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D1_order20/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D6_order5/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D6_order5/', order=5)
    # SibMI experiments with distortion and sibMI budget
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D5_order20_I0.4/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D5_order20_I0.4/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D5_order20_I0.45/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D5_order20_I0.45/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D5_order20_I0.5/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D5_order20_I0.5/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D5_order20_I0.55/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D5_order20_I0.55/', order=20)
    #plot_synth_perf(data=np.load('/home/rxiao/data/synthetic/compare_theoretic/sibMI_privacy_checkpoints1_1_D3.5_order20_I0.2/synth_trainstats.npz'), savedir='synth/SibMI_dist/1_1_D3.5_order20_I0.2/', order=20)
    
    #Produce scatter plots of achieveable levels of distortion vs sibMI, and sibMI vs sec_accuracy
    #plot_synthsib_scatter()
    #plot_synthsib_scatter(datadir='/home/rxiao/data/synthetic/new_exp/', savedir='synth/SibMI_dist/new_scatter/', dataset='synth')
    #plot_synthsib_scatter(datadir='/home/rxiao/data/mnist/remote_exp/', savedir='mnist/SibMI_dist/scatter/', dataset='MNIST')
    #plot_synth_sibvsKL()
    #plot_synth_compare_affine()


if __name__=='__main__':
    main()
