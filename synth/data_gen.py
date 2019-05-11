import numpy as np
import os
import pdb
import sympy
import scipy, scipy.optimize, scipy.stats
import prettytensor as pt
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def sample_x(N, mu1, mu2):
    '''Generate N samples of x by sampling c and x conditioned on c, 
    if c=0, sample from N(mu1, I)
    if c=1, sample from N(mu2, I)
        input: N number of samples
               mu1, np vector
               mu2, np vector
    '''
    assert(len(mu1)==len(mu2))
    dim = len(mu1)
    cs = np.random.choice([0, 1], N)
    mu1s = np.random.multivariate_normal(mu1, np.eye(dim), N)
    mu2s = np.random.multivariate_normal(mu2, np.eye(dim), N)
    pdb.set_trace()
    xs = np.squeeze(mu1s) * cs + np.squeeze(mu2s) * (1-cs)
    return xs, cs

def sample_weighted_x(N, mus):
    '''Generate N samples of x by sampling c from an even prior and x conditioned on c
    input: N number of samples
           mus: array of means
    output: xs array of [NxM] float
            cs vector of class labels [Nx1]
    '''
    dim = len(mus[0])
    for mu in mus:
        if len(mu) != dim:
            return None
    cs = np.random.choice(np.arange(0,dim), N)
    xs = []
    for c in cs:
        xs.extend(np.random.multivariate_normal(mus[c], np.eye(dim)/dim, 1))

    return xs, cs
    
def gen_1d_sample(workdir='/home/rxiao/data/synthetic/'):
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    xs, cs = sample_x(25000, [1], [-1])
    np.savez(workdir+'1d2gaussian_1_m1', c=cs, x=xs)

def gen_1d_weighted_sample(workdir='/home/rxiao/data/synthetic/'):
    """ Generate 1d samples of Gaussians weighted by a given prior probability
    """

def gen_weighted_sample(workdir='/home/rxiao/data/synthetic_weighted/', num_class=10, N=15000):
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    pdb.set_trace()
    # Distribute mus as all 1 except for -1 in one direction for each class
    mus = 2 * (1 - np.eye(num_class)) - 1
    xs, cs = sample_weighted_x(N, mus)
    xs = np.reshape(xs, [N, num_class])
    np.savez(workdir+'weightedgaussian', c=cs, x=xs)

def gen_multi_gauss(N, D1, D2, workdir='/home/rxiao/data/synthetic/'):
    ''' Generate N samples from a multi-dimensional Gausssian distribution
        based on a uniform distribution of binary classes(0,1) in the joint distribution D1, D2
        D1, D2: scalars indicating dimensions in regular/private task
        N: scalar indicating number of samples
    '''
    ys = np.random.choice(np.arange(0, D1), N)
    cs = np.random.choice(np.arange(0, D2), N)
    xs = []

    pdb.set_trace()
    for i in np.arange(N):
        muy = np.zeros(D1)
        muy[ys[i]] = 1
        muc = np.zeros(D2)
        muc[cs[i]] = 1
        xs.append(np.random.multivariate_normal(np.concatenate((muy, muc)), np.eye(D1+D2)))

    np.savez(workdir+'_'.join(['twotaskgaussian', str(D1), str(D2)]), y=ys, c=cs, x=xs)
    print("Saved two task Gaussian data")

def gen_4_gauss(N, workdir='/home/rxiao/data/synthetic/'):
    ''' Generate N samples from 4 conditional Gaussian distributions based on 2 binary class 
        regular task has 2 classes, private task has 2 classes
        N: scalar for number of samples
    '''
    D1 = 2
    D2 = 2
    ys = np.random.choice(np.arange(0, D1), N)
    cs = np.random.choice(np.arange(0, D2), N)
    xs = []

    pdb.set_trace()
    for i in np.arange(N):
        if ys[i]==0 and cs[i]==0:
            mu = 0
        if ys[i]==0 and cs[i]==1:
            mu = 2
        if ys[i]==1 and cs[i]==0:
            mu = 1
        if ys[i]==1 and cs[i]==1:
            mu = 3
        xs.append(np.random.normal(mu, 1))
    xs = np.array(xs).reshape((N, 1))
    np.savez(workdir+'_'.join(['1Dtwotaskgaussian', str(D1), str(D2), '0213']), y=ys, c=cs, x=xs)
    print("Saved 1D two task Gaussian data to %s" % (workdir))

def calc_integral_2_gauss():
    """ calculate symbolic integration for the continuous Gaussian case of 2 conditional Gaussian classes
        For each term of Sibson mutual information of a specified order alpha, we have [\int_z(p(c|z)^a p(z))]^(1/a)
        from the total Sibson MI a/(a-1) log(sum_c [\int_z(p(c|z)^a p(z))]^(1/a))
    """
    z = sympy.Symbol('z')
    #a = sympy.Symbol('a')
    a = 10
    ptilde = sympy.Symbol('ptilde')
    mu0 = sympy.Symbol('mu0')
    mu1 = sympy.Symbol('mu1')
    beta0 = sympy.Symbol('beta0')
    beta1 = sympy.Symbol('beta1')
    sigma = sympy.Symbol('sigma')
    A = sympy.exp(-0.5*((mu1+beta1)/sigma)**2 + 0.5*((mu0+beta0)/sigma)**2)
    B = 0.5*(mu1+beta1 - (mu0+beta0)) / (2*sigma)
    f = (1+((1-ptilde)/ptilde)*A*sympy.exp(B*z))**(-a) * (ptilde*sympy.exp(-0.5*(z-((mu0+beta0) / sigma)**2)) + (1-ptilde)*sympy.exp(-0.5*(z-((mu1+beta1) / sigma)**2)))
    #pdb.set_trace()
    print("Integrating...")
    res = sympy.integrate(f, z)
    print(res)
    file = open("symint.txt", "w")
    file.write(str(res))
    file.close()
    print("Done integrating")
    return

def twogauss_sibMI_theoretic_loss(x, mu0=-3, mu1=3, sigma0=1, sigma1=1, ptilde=0.5, alpha=20):
    """ Implement the loss function based on calculating the upper bound
        of the Sibson MI I(Z;C) for a binary Gaussian distribution Z that
        is the result of an affine transformation of the data X conditional 
        on C.
        args:
        mu0, mu1: means of the conditional X distributions
        sigma0, sigma1: covariances of conditional x distributions
        beta0, beta1: Parameters of the affine distribution Z = X + beta0*(1-C) + beta1*C
        ptilde: prior probability P(C=0)
        D: distortion budget (beta0**2 + beta1**2 <= D)
        alpha: order of the sibson mutual information

        returns:
        an upper bound on Sibson mutual information as a function of the arguments
    """
    # Expand and unpack arguments
    beta0 = x[0]
    beta1 = x[1]
    #mu0, mu1, sigma0, sigma1, ptilde, alpha = args

    mun0 = mu0 + beta0
    mun1 = mu1 + beta1
    #mup0 = min([mun0, mun1])
    #mup1 = max([mun0, mun1])
    mup0 = mun0
    mup1 = mun1
    z0 = (np.log((1-ptilde)/ptilde) + (mup0**2 - mup1**2)/(2*sigma0**2)) * (sigma0**2 / (mup0 - mup1))
    z1 = (np.log(ptilde/(1-ptilde)) + (mup1**2 - mup0**2)/(2*sigma0**2)) * (sigma0**2 / (mup1 - mup0))
    def Q(x):
        return 0.5 * scipy.special.erfc(x/(2.0**0.5))

    if mup1 < mup0:
        h0 = ((1-ptilde)*np.exp((mup0**2 - mup1**2)/(2*sigma0**2))/ptilde)**(-alpha) * np.exp(((mup0 - alpha * (mup1 - mup0))**2 - mup0**2)/(2*sigma0**2)) * (1 - Q((z0 - (mup0 + alpha*(mup0 - mup1)))/sigma0)) + Q((z0 - mup0)/sigma0)
        h1 = ((1-ptilde)*np.exp((mup0**2 - mup1**2)/(2*sigma0**2))/ptilde)**(-alpha) * np.exp(((mup1 - alpha * (mup1 - mup0))**2 - mup0**2)/(2*sigma0**2)) * (1 - Q((z0 - (mup1 + alpha*(mup0 - mup1)))/sigma0)) + Q((z0 - mup1)/sigma0)
        h2 = (ptilde*np.exp((mup1**2 - mup0**2)/(2*sigma0**2))/(1-ptilde))**(-alpha) * np.exp(((mup0 - alpha * (mup0 - mup1))**2 - mup1**2)/(2*sigma0**2)) * Q((z1 - (mup0 + alpha*(mup1 - mup0)))/sigma0) + (1 - Q((z1 - mup0)/sigma0))
        h3 = (ptilde*np.exp((mup1**2 - mup0**2)/(2*sigma0**2))/(1-ptilde))**(-alpha) * np.exp(((mup1 - alpha * (mup0 - mup1))**2 - mup1**2)/(2*sigma0**2)) * Q((z1 - (mup1 + alpha*(mup1 - mup0)))/sigma0) + (1 - Q((z1 - mup1)/sigma0))

    L = (alpha/(alpha - 1)) * np.log((ptilde * h0 + (1-ptilde) * h1)**(1/alpha) + (ptilde * h2 + (1-ptilde) * h3)**(1/alpha))
    return L

def twogauss_sibMI_upperbound(x, mu0=-3.0, mu1=3.0, sigma0=1.0, sigma1=1.0, ptilde=0.5, alpha=5):
    """ Use approximation of (1+Aexp(Bz))**(-alpha) < (Aexp(Bz))**(-alpha)
    """
    #beta0 = x[0]
    #beta1 = x[1]
    #mun0 = mu0 + beta0
    #mun1 = mu1 + beta1
    mun0 = mu0 + x[0]
    mun1 = mu1 + x[1]

    A0 = ((1.0 - ptilde)/ptilde) * np.exp((mun0**2 - mun1**2)/(2*sigma0**2))
    B0 = (mun1 - mun0)/sigma0**2
    A1 = (ptilde/(1.0 - ptilde)) * np.exp((mun1**2 - mun0**2)/(2*sigma0**2))
    B1 = (mun0 - mun1)/sigma0**2
    #pdb.set_trace()
    #h0 = A0**(-alpha) * np.exp(-alpha * B0 * mun0) * np.exp(0.5 * (sigma0**2) * (alpha**2) * (B0**2))
    #h1 = A0**(-alpha) * np.exp(-alpha * B0 * mun1) * np.exp(0.5 * (sigma0**2) * (alpha**2) * (B0**2))
    #h2 = A1**(-alpha) * np.exp(-alpha * B1 * mun0) * np.exp(0.5 * (sigma0**2) * (alpha**2) * (B1**2))
    #h3 = A1**(-alpha) * np.exp(-alpha * B1 * mun1) * np.exp(0.5 * (sigma0**2) * (alpha**2) * (B1**2))

    #alternative bound
    d = (mun1 - mun0)/sigma0
    h0 = np.exp((alpha**2 + alpha) * 0.5 * d**2) * ((1.0 - ptilde) / ptilde)**(-alpha)
    h1 = np.exp((alpha**2 - alpha) * 0.5 * d**2) * ((1.0 - ptilde) / ptilde)**(-alpha)
    h2 = np.exp((alpha**2 + alpha) * 0.5 * d**2) * (ptilde / (1.0 - ptilde))**(-alpha)
    h3 = np.exp((alpha**2 - alpha) * 0.5 * d**2) * (ptilde / (1.0 - ptilde))**(-alpha)
    
    L = (alpha/(alpha - 1)) * np.log((ptilde * h0 + (1-ptilde) * h1)**(1.0/alpha) + (ptilde * h2 + (1-ptilde) * h3)**(1.0/alpha))

    return L


def twogauss_sibMI_loss_tf():
    MAXEPOCH = 1000
    LEARNRATE = 1e-3
    beta0 = tf.Variable(0.0)
    beta1 = tf.Variable(0.0)
    rou_tensor = tf.placeholder(tf.float32)
    mu0 = -3
    mu1 = 3
    sigma0 = 1
    sigma1 = 1
    ptilde = 0.5
    D = 9
    alpha = 5
    mun0 = mu0 + beta0
    mun1 = mu1 + beta1
    mup0 = tf.minimum(mun0, mun1)
    mup1 = tf.maximum(mun0, mun1)
    
    #z0 = (np.log((1-ptilde)/ptilde) + (mup0**2 - mup1**2)/(2*sigma0**2)) * (sigma0**2 / (mup0 - mup1))
    #z1 = (np.log(ptilde/(1-ptilde)) + (mup1**2 - mup0**2)/(2*sigma0**2)) * (sigma0**2 / (mup1 - mup0))
    #def Q(x):
    #    return 0.5 * tf.erfc(x/(2.0**0.5))
    #h0 = ((1-ptilde)*tf.exp((mup0**2 - mup1**2)/(2*sigma0**2))/ptilde)**(-alpha) * tf.exp(((mup0 + alpha * (mup0 - mup1))**2 - mup0**2)/(2*sigma0**2)) * (1 - Q((z0 - (mup0 + alpha*(mup0 - mup1)))/sigma0)) + Q((z0 - mup0)/sigma0)
    #h1 = ((1-ptilde)*tf.exp((mup0**2 - mup1**2)/(2*sigma0**2))/ptilde)**(-alpha) * tf.exp(((mup1 + alpha * (mup0 - mup1))**2 - mup0**2)/(2*sigma0**2)) * (1 - Q((z0 - (mup1 + alpha*(mup0 - mup1)))/sigma0)) + Q((z0 - mup1)/sigma0)
    #h2 = (ptilde*tf.exp((mup1**2 - mup0**2)/(2*sigma0**2))/(1-ptilde))**(-alpha) * tf.exp(((mup0 + alpha * (mup1 - mup0))**2 - mup1**2)/(2*sigma0**2)) * Q((z1 - (mup0 + alpha*(mup1 - mup0)))/sigma0) + (1 - Q((z1 - mup0)/sigma0))
    #h3 = (ptilde*tf.exp((mup1**2 - mup0**2)/(2*sigma0**2))/(1-ptilde))**(-alpha) * tf.exp(((mup1 + alpha * (mup1 - mup0))**2 - mup1**2)/(2*sigma0**2)) * Q((z1 - (mup1 + alpha*(mup1 - mup0)))/sigma0) + (1 - Q((z1 - mup1)/sigma0))

    #L = (alpha/(alpha - 1)) * tf.log((ptilde * h0 + (1-ptilde) * h1)**(1/alpha) + (ptilde * h2 + (1-ptilde) * h3)**(1/alpha))
    
    #alternative bound
    d = (mun1 - mun0)/sigma0
    h0 = tf.exp((alpha**2 + alpha) * 0.5 * d**2) * ((1.0 - ptilde) / ptilde)**(-alpha)
    h1 = tf.exp((alpha**2 - alpha) * 0.5 * d**2) * ((1.0 - ptilde) / ptilde)**(-alpha)
    h2 = tf.exp((alpha**2 + alpha) * 0.5 * d**2) * (ptilde / (1.0 - ptilde))**(-alpha)
    h3 = tf.exp((alpha**2 - alpha) * 0.5 * d**2) * (ptilde / (1.0 - ptilde))**(-alpha)
    
    L = (alpha/(alpha - 1)) * tf.log((ptilde * h0 + (1-ptilde) * h1)**(1.0/alpha) + (ptilde * h2 + (1-ptilde) * h3)**(1.0/alpha))


    totalloss = L + rou_tensor * tf.maximum(beta0**2 + beta1**2 - D, 0)

    optimizer = tf.train.AdamOptimizer(LEARNRATE, epsilon=1.0)
    trainstep = optimizer.minimize(totalloss)
    #trainstep = pt.apply_optimizer(optimizer, losses=[totalloss], regularize=True, include_marked=True, var_list=tf.trainable_variables())

    # Logging matrices
    loss_val = np.zeros(MAXEPOCH)
    sibMI_val = np.zeros(MAXEPOCH)
    beta0_val = np.zeros(MAXEPOCH)
    beta1_val = np.zeros(MAXEPOCH)
    rou_values = np.linspace(0, 5000, MAXEPOCH)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Config session for memory
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.log_device_placement=False

    sess = tf.Session(config=config)
    sess.run(init)

    for epoch in range(MAXEPOCH):
        feeds = {rou_tensor: rou_values[epoch]}
        beta0tmp, beta1tmp, sibMItmp, losstmp, _ = sess.run([beta0, beta1, L, totalloss, trainstep], feeds)
        loss_val[epoch] = losstmp
        sibMI_val[epoch] = sibMItmp
        beta0_val[epoch] = beta0tmp
        beta1_val[epoch] = beta1tmp
    pdb.set_trace()
    print(sess.run([beta0, beta1]))
    np.savez(os.path.join(os.getcwd(), 'synth_opt_stats'), loss_val=loss_val,
            sibMI_val = sibMI_val,
            beta0=beta0_val,
            beta1=beta1_val)

def twogauss_sibMI_theoretic_opt():
    """ Implement optimization for the minimization of the upper bound on Sibson MI
        based on a binary mixture of Gaussian data and affine transformations.
    """
    mu0 = -3
    mu1 = 3
    sigma0 = 1
    sigma1 = 1
    ptilde = 0.5
    D = 9
    alpha = 20
    args = (mu0, mu1, sigma0, sigma1, ptilde, D, alpha)
    init = [0, 0]
    def dist_fn(x):
        return D - (ptilde * x[0]**2 + (1-ptilde) * x[1]**2)
    # Inequality constraints are defined such that F(x) >= 0
    dist_constraint = {'type':'ineq', 'fun': dist_fn}
    Ds = np.linspace(0.5, 10, 20)
    xs = []
    for D in Ds:
        #opt_res = scipy.optimize.minimize(twogauss_sibMI_theoretic_loss, init, options={'disp': True}, constraints=dist_constraint)
        opt_res = scipy.optimize.minimize(twogauss_sibMI_upperbound, init, options={'maxiter': 50, 'disp': True}, constraints=dist_constraint)
        if opt_res.success:
            print("Optimization successful")
            print("Final params: %s" % opt_res.x)
            xs.append(opt_res.x)
        else:
            print("Optimization failed due to %s" % opt_res.message)
    # Calculate theoretic MAP accuracy
    betas = np.concatenate(xs).reshape((len(Ds),2))
    means = np.array((-3, 3)) + xs
    newphis = scipy.stats.norm.cdf(means[:,1])
    #phis = 0.5 * (1 + scipy.special.erf(means[:,1])/np.sqrt(2.0))
    pdb.set_trace()
    np.savez("synth/theoretic_upper_bound_alpha5", Ds=Ds,xs=xs, phis=newphis)
    return

def twogauss_sibMI_theoretic_plot():
    """ Plot the loss for ranges of beta0 and beta1 for affine transformations
        on a binary mixture of Gaussian, where the loss is the upper bound on 
        sibson MI
    """
    mu0 = -3
    mu1 = 3
    sigma0 = 1
    sigma1 = 1
    ptilde = 0.5
    D = 9
    alpha = 2
    Npts = 200

    beta0s = np.linspace(-2, 2, Npts)
    beta1s = np.linspace(-2, 2, Npts)
    L2 = np.zeros((Npts, Npts))
    L10 = np.zeros((Npts, Npts))
    pdb.set_trace()
    for i in xrange(Npts):
        for j in xrange(Npts):
            #Ls[i,j] = twogauss_sibMI_theoretic_loss([beta0s[i], beta1s[j]], alpha=2)
            L2[i,j] = twogauss_sibMI_upperbound([beta0s[i], beta1s[j]], alpha=2)
            L10[i,j] = twogauss_sibMI_upperbound([beta0s[i], beta1s[j]], alpha=5)

    pdb.set_trace()
    beta0s, beta1s = np.meshgrid(beta0s, beta1s)
    fig = plt.figure(figsize=(16,16))
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(1,1,1,projection='3d')
    p = ax.plot_surface(beta0s, beta1s, L2, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=False, label='alpha=2')
    p = ax.plot_surface(beta0s, beta1s, L10, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=False, label='alpha=5')
    cb = fig.colorbar(p, shrink=0.5)
    
    #ax = fig.add_subplot(1,1,1,projection='3d')
    #p = ax.plot_wireframe(beta0s, beta1s, L2, rstride=4, cstride=4)
    #p = ax.plot_wireframe(beta0s, beta1s, L10, rstride=4, cstride=4)
    #plt.legend()
    plt.show()
    plt.savefig("synth/plot_sibMI_bound.png", bbox_inches='tight')

    return

def twogauss_sibMI_approx_comp():
    """ Compare the approximation of sibson MI with the true sib MI for binary gaussian, with means -3 and 3
        over different values of alpha as an approximate sum over Z as the integral
        also compute the maximal information leakage as a comparison
    """
    def Q(x):
        return 0.5 * scipy.special.erfc(x/(2.0**0.5))
    from scipy.stats import norm
    alphas = np.linspace(1.1, 100, 500)
    mu0 = -3
    mu1 = 3
    sigma0 = 1
    sigma1 = 1
    ptilde = 0.5
   
    x0 = np.linspace(norm.ppf(0.001, mu0, sigma0), norm.ppf(0.999, mu0, sigma0), 1000)
    x1 = np.linspace(norm.ppf(0.001, mu1, sigma1), norm.ppf(0.999, mu1, sigma1), 1000)
    x = np.linspace(norm.ppf(0.001, mu0, sigma0), norm.ppf(0.999, mu1, sigma1), 1000) # take 1000 pts from one end of gauss 0 to tail of gauss 1
    y0 = norm.pdf(x, mu0, sigma0)
    y1 = norm.pdf(x, mu1, sigma1)
    #compute the true sibson mutual information as a sum over z
    sibMIs = []
    sibMIapproxs = []
    maxILs = []
    KLs = []
    for alpha in alphas:
        sumz = ((y0**alpha)*ptilde + (y1**alpha)*(1-ptilde))**(1.0/alpha)
        sibMI = (alpha/(alpha - 1)) * np.log(np.sum(sumz))
        sumzapprox = y0 * (ptilde**(1.0/alpha)) * np.maximum(1, ((1-ptilde)/ptilde)**(1.0/alpha) * (y1/y0))
        sibMIapprox = (alpha/(alpha - 1)) * np.log(np.sum(sumzapprox))
        sibMIs.append(sibMI)
        sibMIapproxs.append(sibMIapprox)
        #maxILs.append(np.log(2 * Q((mu0 - mu1)/(2 * sigma0))))
        KL = np.sum(y0 * ptilde * np.log(y0 / (y0 * ptilde + y1 * (1 - ptilde))) + y1 * (1 - ptilde) * np.log(y1 / (y0 * ptilde + y1 * (1 - ptilde))))
        #KLs.append(np.sum(y0 * ptilde * np.log(y0 / ptilde) + y1 * (1 - ptilde) * np.log(y1 / (1 - ptilde))))
        KLs.append(KL)

    # Plot the approx and full value of sibMI over alphas and amp
    plt.plot(alphas, sibMIs, '-d', markersize=2, label="Sibson MI")
    plt.plot(alphas, sibMIapproxs, '-o', markersize=2, label="Sibson MI approximation")
    plt.plot(alphas, maxILs, '--', markersize=2, label="Maximal Info Leakage")
    plt.plot(alphas, KLs, '-o', markersize=2, label="MI")
    plt.xlabel("Order alpha")
    plt.ylabel("Sibson MI")
    plt.title("Sibson MI and approximation vs alpha")
    plt.legend()
    plt.savefig("synth/sibMI_sibMIapprox_alpha", bbox_inches='tight')
    plt.show()
    plt.close()
   
    pdb.set_trace()
    np.savez("synth/sibMI_sibMIapprox_alpha.npz", alpha=alphas, sibMI=sibMIs, sibMIapprox=sibMIapproxs)
    # plot over different amplitudes of means
    alpha = 20
    amp = np.linspace(1, 4, 100)
    sigma0 = 1
    sigma1 = 1
    ptilde = 0.5
    sibMIs = []
    sibMIapproxs = []
    maxILs = []
    for i in xrange(len(amp)):
        mu0 = -amp[i]
        mu1 = amp[i]
        x0 = np.linspace(norm.ppf(0.001, mu0, sigma0), norm.ppf(0.999, mu0, sigma0), 1000)
        x1 = np.linspace(norm.ppf(0.001, mu1, sigma1), norm.ppf(0.999, mu1, sigma1), 1000)
        x = np.linspace(norm.ppf(0.001, mu0, sigma0), norm.ppf(0.999, mu1, sigma1), 1000) # take 1000 pts from one end of gauss 0 to tail of gauss 1
        y0 = norm.pdf(x, mu0, sigma0)
        y1 = norm.pdf(x, mu1, sigma1)
        sumz = ((y0**alpha)*ptilde + (y1**alpha)*(1-ptilde))**(1.0/alpha)
        sibMI = (alpha/(alpha - 1)) * np.log(np.sum(sumz))
        sumzapprox = y0 * (ptilde**(1.0/alpha)) * np.maximum(1, ((1-ptilde)/ptilde)**(1.0/alpha) * (y1/y0))
        sibMIapprox = (alpha/(alpha - 1)) * np.log(np.sum(sumzapprox))
        if mu0 < mu1:
            maxIL = np.log(2 * Q((mu0 - mu1) / (2 * sigma0)))
        else:
            maxIL = np.log(2 * Q((mu1 - mu0) / (2 * sigma0)))
        sibMIs.append(sibMI)
        sibMIapproxs.append(sibMIapprox)
        maxILs.append(maxIL)
    pdb.set_trace()
    np.savez("synth/sibMI_sibMIapprox_amp.npz", amp=amp, sibMI=sibMIs, sibMIapprox=sibMIapproxs)
    
    # Plot the approx and full value of sibMI over alphas and amp
    plt.plot(amp, sibMIs, '-d', markersize=4, label="Sibson MI")
    plt.plot(amp, sibMIapproxs, '-o', markersize=4, label="Sibson MI approximation")
    plt.plot(amp, maxILs, '--', markersize=4, label="Maximal Info Leakage")
    plt.xlabel("Amplitude of mu")
    plt.ylabel("Sibson MI")
    plt.title("Sibson MI and approximation vs amp of means(mu and -mu)")
    plt.legend()
    plt.savefig("synth/sibMI_sibMIapprox_amp", bbox_inches='tight')
    plt.show()
    plt.close()
 
    return

def KL_vs_d_ptilde():
    """ Plot the accuracy of MAP for binary gaussian over a span of d and ptilde
    """
    def Q(x):
        return 0.5 * scipy.special.erfc(x/(2.0**0.5))
    N = 1000
    ds = np.linspace(0.001, 100, N)
    ptildes = np.linspace(0.01, 0.99, N)
    MAPaccuracy = np.zeros((N, N))
    MAPacc_dd = np.zeros((N,N))
    for i in xrange(N):
        for j in xrange(N):
            d = ds[i]
            ptilde = ptildes[j]
            MAPaccuracy[i,j] = ptilde*Q(-d/2.0 + np.log((1.0 - ptilde)/ptilde)/d) + (1.0-ptilde)*Q(-d/2.0 - np.log((1.0 - ptilde)/ptilde)/d)
            MAPacc_dd[i,j] = (ptilde/np.sqrt(2 * math.pi)) * np.exp(0.5*(d/2. - np.log((1.0 - ptilde)/ptilde)/d)) * (-0.5 - np.log((1.0 - ptilde)/ptilde)/d**2) + \
                             ((1.0 - ptilde)/np.sqrt(2 * math.pi)) * np.exp(0.5*(d/2. + np.log((1.0 - ptilde)/ptilde)/d)) * (-0.5 + np.log((1.0 - ptilde)/ptilde)/d**2)
   
    plt.plot(ds, MAPaccuracy[:, 100], label="ptilde=%s"%(ptildes[100]))
    plt.plot(ds, MAPaccuracy[:, 300], label="ptilde=%s"%(ptildes[300]))
    plt.plot(ds, MAPaccuracy[:, 500], label="ptilde=%s"%(ptildes[500]))
    plt.plot(ds, MAPaccuracy[:, 700], label="ptilde=%s"%(ptildes[700]))
    plt.plot(ds, MAPaccuracy[:, 900], label="ptilde=%s"%(ptildes[900]))
    plt.legend()
    plt.show()
    plt.close()
    plt.plot(ds, MAPacc_dd[:, 100], label="ptilde=%s"%(ptildes[100]))
    plt.plot(ds, MAPacc_dd[:, 300], label="ptilde=%s"%(ptildes[300]))
    plt.plot(ds, MAPacc_dd[:, 500], label="ptilde=%s"%(ptildes[500]))
    plt.plot(ds, MAPacc_dd[:, 700], label="ptilde=%s"%(ptildes[700]))
    plt.plot(ds, MAPacc_dd[:, 900], label="ptilde=%s"%(ptildes[900]))
    plt.legend()
    plt.show()
    plt.close()


    #mesh grid plots
    ds, ptildes = np.meshgrid(ds, ptildes)
    fig = plt.figure(figsize=(16,16))
    ax = fig.gca(projection='3d')
    ax = fig.add_subplot(1,1,1,projection='3d')
    #p = ax.plot_surface(ds, ptildes, MAPaccuracy, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=False, label='MAPaccuracy')
    #cb = fig.colorbar(p, shrink=0.5)
    #ax = fig.add_subplot(1,1,1,projection='3d')
    p = ax.plot_wireframe(ds, ptildes, MAPaccuracy, rstride=4, cstride=4)
    ax.set_xlabel("D")
    ax.set_ylabel("Ptilde")
    ax.set_zlabel("MAP accuracy")
 
    #plt.show()
    #plt.savefig("synth/plotMAPacc_vs_d_ptilde.png", bbox_inches='tight')
    mapaccminargs = [np.argmin(MAPaccuracy[:, i]) for i in xrange(N)]
    mapaccmins = [MAPaccuracy[mapaccminargs[i], i] for i in xrange(N)]
    pdb.set_trace()

    def acc(x, *args):
        d = x
        ptilde = args
        #return ptilde*Q(-d/2.0 + np.log((1.0 - ptilde)/ptilde)/d) + (1.0-ptilde)*Q(-d/2.0 - np.log((1.0 - ptilde)/ptilde)/d)
        return ptilde * 0.5 * scipy.special.erfc((-d/2.0 + np.log((1.0 - ptilde)/ptilde)/d)/(2.0**0.5)) + (1.0-ptilde) * 0.5 * scipy.special.erfc((-d/2.0 - np.log((1.0 - ptilde)/ptilde)/d)/(2.0**0.5))

    def dist_fn(x):
        return x
    accs = np.zeros(N)
    # Inequality constraints are defined such that F(x) >= 0
    dist_constraint = {'type':'ineq', 'fun': dist_fn}
    for i in xrange(N):
        opt_res = scipy.optimize.minimize(acc, 100.0, args=(ptildes[i]), constraints=dist_constraint)
        if opt_res.success:
            print("Optimization successful")
            print("Final params: %s" % opt_res.x)
            accs[i] = opt_res.x
        else:
            print("Optimization failed due to %s" % opt_res.message)
    pdb.set_trace()
 
    return
    
def simple_opt():
    """Example to use scipy optimization"""
    def f(x):
        return -np.exp(-(x-0.7)**2)
    opt_res = scipy.optimize.minimize(f, 0)
    if opt_res.success:
        print("Optimized example")
        print("Final params: %s" % opt_res.x)
        #pdb.set_trace()
    return


if __name__ == '__main__':
    #gen_weighted_sample()
    #gen_1d_sample()
    #gen_multi_gauss(15000, 2, 2)
    #gen_4_gauss(15000)

    #calc_integral_2_gauss()
    #simple_opt()
    #twogauss_sibMI_theoretic_opt()
    #twogauss_sibMI_loss_tf()
    #twogauss_sibMI_theoretic_plot()
    twogauss_sibMI_approx_comp()
    #KL_vs_d_ptilde()
