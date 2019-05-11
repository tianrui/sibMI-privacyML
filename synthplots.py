"""Synthetic plots of binary distributions and the Sibson mutual information
"""
import os
import numpy as np
import operator as op
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import plot
import pdb

def sibsonMI_approx(pz, pcz, alpha):
    '''Calculate the Sibson MI given a vector of pz probabilities and 
       a matrix of pc|z probabilites, and the alpha coefficient, according to
       SIB(Z;C)=sum_c (sum_z(pz * (pc|z=z)^alpha)^(1/alpha))
       pz binary len 2 vector
       pcz binary len 4 vector organized for conditional distribution as 
       (0,0), (0, 1), (1, 0), (1, 1) for c, z pairs
    '''
    pdb.set_trace()
    sumz = np.array([np.sum(pz * (pcz[[0, 1]])**alpha), np.sum(pz * (pcz[[2,3]])**alpha)])
    sibMI = alpha/(alpha-1) * np.log(np.sum(sumz**(1.0/alpha)))
    return sibMI

def KL_approx(pz, pcz, pc):
    '''
        Assuming that pcz is a binary len 4 vector as a conditional distribution
       (0,0), (0, 1), (1, 0), (1, 1) for c, z pairs
    '''
    KL = np.sum(pz[0] * pcz[[0,2]] * np.log(pcz[[0,2]]/pc)) + pz[1] * np.sum(pcz[[1,3]] * np.log(pcz[[1,3]]/pc))
    #pdb.set_trace()
    return KL

def synthesize_sibMI_uniform(savedir='/home/rxiao/code/dvib/synth'):
    ''' Create synthetic distribution of pz based on uniform [0,1], then pc|z as a binary distribution p(c=0|z)=z
        and calculate the Sibson MI and KL divergence
    '''
    def sibsonMI_approx_uniform(pcz, alpha):
        sibMI = alpha/(alpha-1.0) * np.log((np.average(pcz**alpha))**(1.0/alpha) + (np.average((1.0-pcz)**alpha))**(1.0/alpha))
        return sibMI
    def KL_approx_uniform(pcz, pc):
        KL = np.average(pcz * np.log(pcz/pc) + (1.0-pcz) * np.log((1.0-pcz)/(1.0-pc)))
        return KL
    N=10000
    ind = N/2
    #Z drawn from uniform distribution
    pz = np.random.uniform(0.001, 1.0, N)
    #P(C=0|Z=z) is probability that C is in class 0
    #with the probability being z
    pcz = pz
    #P(C=0) calculated by averaging P(C=0|Z=z) over the samples of z
    pc = np.average(pcz)
    alpha = np.linspace(1.0001, 40, N)
    sibMI_alpha = [sibsonMI_approx_uniform(pcz, alphai) for alphai in alpha]
    KL_alpha = [KL_approx_uniform(pcz, pc) for alphai in alpha]
    pdb.set_trace()
    np.savez(os.path.join(savedir, 'sibsonMI_synth_uniform'),
                        alpha=alpha,
                        pz=pz,
                        pcz=pcz,
                        sibMI_alpha=sibMI_alpha,
                        KL_alpha=KL_alpha)
    print("Saved uniform generation and Sibson MI calculation")


def synthesize_sibMI(savedir='/home/rxiao/code/dvib/synth'):
    '''Create some synthetic distributions of pz, pcz and plot based on
       varying alpha coefficients
       pz ranges from 0 - 1 as a binary distribution
       Assuming that pcz is a binary len 4 vector as a conditional distribution
       (0,0), (0, 1), (1, 0), (1, 1) for c, z pairs
       
    '''
    N=1000
    #pzN = np.linspace(0.1, 0.9, N)
#   Generate pz
    #pz = np.array([pzN, 1-pzN])
#   Generate pc,z
    pcz0 = np.reshape(np.linspace(0.1, 0.9, N), (1,N))
    pcz1 = np.tile(np.array(1.0 - pcz0)/3, (3, 1))
    pcz = np.concatenate((pcz0, pcz1))
    pcz = pcz.T
#   Convert to conditional distribution
    pcz_cond = np.zeros((N,4))
    pcz_cond[:,0] = pcz[:,0]/(pcz[:,0] + pcz[:,1])
    pcz_cond[:,1] = pcz[:,1]/(pcz[:,0] + pcz[:,1])
    pcz_cond[:,2] = pcz[:,2]/(pcz[:,2] + pcz[:,3])
    pcz_cond[:,3] = pcz[:,3]/(pcz[:,2] + pcz[:,3])
#   Calculate marginal pz
    pz = np.array([pcz[:,0]+pcz[:,2], pcz[:,1]+pcz[:,3]])
    pz = pz.T
#   Calculate marginal pc
    pc = np.array([pcz[:,0]+pcz[:,1], pcz[:,2]+pcz[:,3]])
    pc = pc.T
    pcz = pcz_cond
#   Generate qcz as a noisy version
    noise = 1e-1 * np.random.normal(0,1)
    qcz = np.zeros((N,4))
    qcz[:,0] = pcz[:,0] + noise
    qcz[:,1] = pcz[:,1] - noise
    qcz[:,2] = pcz[:,2] + noise
    qcz[:,3] = pcz[:,3] - noise
    alpha = np.linspace(1.0001, 2, N)
   
    ind=N/2
    sibMI_pz = [sibsonMI_approx(pzi, pcz[ind], alpha[1]) for pzi in pz]
    sibMI_pcz = [sibsonMI_approx(pz[ind], pczi, alpha[1]) for pczi in pcz]
    sibMI_alpha = [sibsonMI_approx(pz[ind], pcz[ind], alphai) for alphai in alpha]
    sibMIq_alpha = [sibsonMI_approx(pz[ind], qcz[ind], alphai) for alphai in alpha]
    KL_alpha = [KL_approx(pz[ind], pcz[ind], pc[ind]) for alphai in alpha]
    pdb.set_trace()
    np.savez(os.path.join(savedir, 'sibsonMI_synth'), 
                      pz=pz,
                      sibMI_pz=sibMI_pz,
                      pcz=pcz,
                      sibMI_pcz=sibMI_pcz,
                      alpha=alpha,
                      sibMI_alpha=sibMI_alpha,
                      sibMIq_alpha=sibMIq_alpha,
                      KL_alpha=KL_alpha)
    print("Saved synthetic sibson MI plot points")

    return

def synthesize_compare(savedir='/home/rxiao/code/dvib/synth'):
    '''Compare the bounds for (sum_z pz * pcz^alpha)^(1/alpha)
       and sum_z (p_z^1/alpha * pcz)
    '''
    N=1000
    pzN = np.linspace(1.0/N, 1, N)
    pz = np.array([pzN, 1-pzN])
    pcz0 = np.reshape(np.linspace(1.0/N, 1, N), (1,N))
    pcz1 = np.tile(np.array(1.0 - pcz0)/3, (3, 1))
    pcz = np.concatenate((pcz0, pcz1))
    alpha = np.arange(1, 20, 1)
    
    pz = pz.T
    pcz = pcz.T
    def comp_sib(pz, pcz, alpha):
        sumz_out = np.array([np.sum(pz * (pcz[:len(pz)])**alpha), np.sum(pz * (pcz[len(pz):])**alpha)])
        sumz_outer = np.sum(sumz_out**(1.0/alpha))
        sumz_in = np.array([np.sum(pz**(1.0/alpha) * pcz[:len(pz)]), np.sum(pz**(1.0/alpha) * pcz[len(pz):])])
        sumz_inner = np.sum(sumz_in)
        return sumz_outer, sumz_inner
    ind = N/2 
    listpz = zip(*[comp_sib(pzi, pcz[ind], alpha[1]) for pzi in pz])
    pdb.set_trace()
    sumz_out_pz, sumz_in_pz = [list(i) for i in listpz]
    listpcz = zip(*[comp_sib(pz[ind], pczi, alpha[3]) for pczi in pcz])
    sumz_out_pcz, sumz_in_pcz = [list(i) for i in listpcz]
    listalpha = zip(*[comp_sib(pz[ind], pcz[ind], alphai) for alphai in alpha])
    sumz_out_alpha, sumz_in_alpha = [list(i) for i in listalpha]

    np.savez(os.path.join(savedir, 'sibsonMI_comp'), 
                      pz=pz,
                      sumz_out_pz=sumz_out_pz,
                      sumz_in_pz=sumz_in_pz,
                      pcz=pcz,
                      sumz_out_pcz=sumz_out_pcz,
                      sumz_in_pcz=sumz_in_pcz,
                      alpha=alpha,
                      sumz_out_alpha=sumz_out_alpha,
                      sumz_in_alpha=sumz_in_alpha)
    print("Saved synthetic comparison plot points")
    return

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, linspace(n, n-r, r+1), 1)
    denom = reduce(op.mul, linspace(1, r+1, r+2), 1)
    return numer//denom

def multinomial_approx(xs, alpha, degree=3):
    '''Expand the multinomial given by xs as (x1+x2+...xm)^alpha
       to the degree given
    '''
    return

def plot_simple_sibMI(order=2):
    return
 


if __name__=='__main__':
    #synthesize_sibMI('/home/rxiao/code/dvib/synth')
    #plot.plot_sibson_synth(np.load('/home/rxiao/code/dvib/synth/sibsonMI_synth.npz'), '/home/rxiao/code/dvib/synth/')
    #synthesize_compare()
    #plot.plot_sibson_comp(np.load('/home/rxiao/code/dvib/synth/sibsonMI_comp.npz'), '/home/rxiao/code/dvib/synth/')
    synthesize_sibMI_uniform('/home/rxiao/code/dvib/synth')
    plot.plot_sibson_synth(np.load('/home/rxiao/code/dvib/synth/sibsonMI_synth_uniform.npz'), '/home/rxiao/code/dvib/synth/uniform/')
