from __future__ import absolute_import, division, print_function
import numpy as np
import os
import turtle
from PIL import Image
import matplotlib.pyplot as plt

import pdb

""" This file contains functions and the script for preprocessing 
    the pendigits database from the dynamic 16 representation
    which consists of 8 successive pen strokes on a coordinate 
    system. The digits will be drawn using turtle on a 100x100 grid,
    centered, and downsampled to 20x20 pixels. This is to maintain
    consistency with the previous literature
"""

def load_data(data_dir):
    ''' Load data for pendigits as an array of Nx256
    '''
    data = np.loadtxt(data_dir, delimiter=',')
    return data

def proc_img(strokes, savename="tmpturtle.ps"):
    ''' Process one image from 8 penstrokes
        strokes: length 16 list of coordinate pairs
    '''
    turtle.screensize(120, 120)
    turtle.setworldcoordinates(-10, -10, 110, 110)
    #turtle.bgcolor("white")
    #turtle.pencolor("black")
    turtle.width(20)
    turtle.penup()
    turtle.ht()
    turtle.goto(strokes[0:2])
    turtle.pendown()
    for i in xrange(0, 8):
        turtle.goto(strokes[2*i: 2*i+2])

    turtle.penup()
    #pdb.set_trace()
    saver = turtle.getscreen()
    saver.getcanvas().postscript(file=savename, colormode="gray")
    turtle.bye()
    
    return

def proc_dataset():
    data = load_data("/home/rxiao/data/pendigits/pendigits_dyn_train.csv")
    #doing temporary work with one image
    proc_img(data[309])
    tmpimg = Image.open("tmpturtle.ps")
    pdb.set_trace()
    tmpimg = tmpimg.resize((20, 20), Image.ANTIALIAS)
    tmparray = np.array(tmpimg.convert('L'))
    plt.imshow(tmparray, cmap='gray')
    plt.show()
    
    return

if __name__=="__main__":
    proc_dataset()
