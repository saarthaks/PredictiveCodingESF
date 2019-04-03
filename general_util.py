import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import random

# general util
def compute_rmse(x,x_hat):
    return np.sqrt(np.mean((x-x_hat)**2))

def generate_signal(I,tmin,tmax,dt,fmax=4,amp=1):
    trange = np.arange(0,tmax-tmin,dt)
    tlen = len(trange)
    sig = np.zeros((I,tlen))
    num_signals = 10
    for dim in range(I):
        for j in range(num_signals):
            p = fmax*2*np.pi*(2*np.random.rand()-1)
            s = amp*(2*np.random.rand()-1)
            sig[dim,:] = sig[dim,:] + s*np.sin(p*trange)
        sig[dim,:] = sig[dim,:] - sig[dim,0]
    
    return sig

def filter_signal(c, dt, lam_x):
    x = np.zeros_like(c)
    for t in range(1,c.shape[1]):
        x[:,t] = x[:,t-1] + dt * (c[:,t-1] - lam_x*x[:,t-1])
    
    return x

def plot_reconstruction(ts, x, xh, rmse):
    plt.figure(figsize=(14,6))
    titl = 'RMSE = %.6f' % rmse
    plt.title(titl)
    for j in range(x.shape[0]):
        lbl = 'est dim %d' % j
        plt.plot(ts, x[j,:])
        plt.plot(ts, xh[j,:], label=lbl)
    plt.legend()

def plot_raster(N, T, spktm, rho=0):
    plt.figure(figsize=(14,6))
    frate = np.zeros((N,))
    for n in range(N):
        spikes = spktm[n, np.nonzero(spktm[n,:])]    # fetch all the spike times (remove the zeros)
        frate[n] = spikes.size / float(T)
        plt.scatter(spikes, n*np.ones((1,spikes.size)));     # plot points at spike times at the n-th row

    titl = 'Mean Firing Rate: %.2f' % (np.mean(frate)/(1-rho))
    plt.title(titl)
    plt.xlim(0,1.05*T)
    plt.ylim(-0.5,N+0.5)

def plot_balance(ni, ts, Idiff, ei_var, limits):
    plt.figure(figsize=(14,6))
    titl = "E-I Variance = %.2E" % Decimal(ei_var)
    plt.plot(ts, Idiff[ni,:])
    plt.title(titl)
    plt.ylim(limits)

def compute_cv(N, o):
    cv = []
    for i in range(N):
        tmp = np.where(o[i,:])[0]
        if len(tmp)>1:
            tmp2 = np.insert(tmp,0,tmp[0])
            tmp3 = tmp2[1:]-tmp2[:-1]
            top = np.abs(tmp3[1:]-tmp3[:-1])
            bot = tmp3[1:]+tmp3[:-1]
            cv.append(np.mean(top/bot))
    
    return np.mean(cv)

