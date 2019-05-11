import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import h5py
import pdb

def read_img(data_dir, savedir):
    '''Map FERG data into 50x50 grayscale images,
       map the character label and expression label to numbers
       store as an npz
    '''
    reglabels = []
    privlabels = []
    imgs = []
    expressiondict = {'anger':0,
                      'disgust':1,
                      'fear':2,
                      'joy':3,
                      'neutral':4,
                      'sadness':5,
                      'surprise':6}
    charadict = {'aia':0,
                 'bonnie':1,
                 'jules':2,
                 'malcolm':3,
                 'mery':4,
                 'ray':5}
    for subdirs, dirs, filenames in os.walk(data_dir):
        for name in filenames:
            if name.split('.')[1] == 'png':
                try:
                    #pdb.set_trace()
                    img = Image.open(os.path.join(subdirs,name))
                    tmp = name.split('_')
                    privlabel = charadict[tmp[0]]
                    reglabel = expressiondict[tmp[1]]
                    img.thumbnail((50, 50), Image.ANTIALIAS)
                    imgs.append(np.array(img.convert('L')))
                    reglabels.append(reglabel)
                    privlabels.append(privlabel)
                except IOError:
                    print("Error with opening %s" % name)
    np.savez(savedir+'ferg256.npz', imgs=imgs,
                                    identity=privlabels,
                                    expression=reglabels)
    print("Saved converted FERG data")
    return

if __name__=='__main__':
    read_img('/cad2/ece521s/cuda_libs/data/', '/cad2/ece521s/cuda_libs/data/')
