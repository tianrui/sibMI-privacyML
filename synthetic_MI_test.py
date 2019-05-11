import numpy as np
import os
import pdb
import matplotlib
import matplotlib.pyplot as plt

import losses as dvibloss
import components as dvibcomp

def calc_sibsonMI_approx(pz, pcz, alpha):
    """Calculate the sibson mutual information of order alpha based on 
      alpha/(alpha-1) log (sum_c (sum_z pz pcz^alpha)^(1/alpha))
      pz: array of Nx1
      pcz: array of MxN
    """
    if np.sum(pz) != 1.0 or np.sum(pcz) != 1.0:
       print "Input arguments to sibson MI not valid distributions"
    sumz = np.sum(np.multiply(pz, pcz**alpha), axis=1)
    return alpha/(alpha-1) * np.log(np.sum(sumz^(1.0/alpha)))

def entropy(probs):
    """Calculate regular entropy for a vector of probabilities
    """
    if np.abs(np.sum(probs) - 1.0) > 1e-6:
        print("Entropy calculation failed: inputs not a probability")
        return None

    return np.sum(np.multiply(-np.log(probs), probs))
