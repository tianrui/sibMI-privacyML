import mnist_privacy as mnistpriv
import plot as dvibplot

import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

def visualize_xhat(encode_coef=100):
    datadir = "/cad2/ece521s/cuda_libs/mnist/data/mnist/privacy_checkpoints"+str(encode_coef)+"/vis_x_xhat.npz"
    print(datadir)
    savedir = "/cad2/ece521s/cuda_libs/visuals/mnist/KLprivacy_checkpoints"+str(encode_coef)
    mnistpriv.eval_checkpt(encode_coef, lossmetric="")
    dvibplot.plot_mnist_x_xhat(np.load(datadir), savedir, encode_coef, 4, 3)
    return



if __name__=="__main__":
    #visualize_xhat(70)
    #visualize_xhat(60)
    #visualize_xhat(50)
    visualize_xhat(40)
    #visualize_xhat(100)
    #visualize_xhat(10)
    #visualize_xhat(20)
