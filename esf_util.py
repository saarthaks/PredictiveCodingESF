import numpy as np
import matplotlib.pyplot as plt
import random
from general_util import *

# esf util
def initialize_standard(N, J, G_scale, lam_x, lam_r, mu=0):
    G = np.random.randn(J, N)
    G = G_scale * (G/np.linalg.norm(G, axis=0)[np.newaxis,:])
    Th = float(0.5 * (G_scale**2 + mu))
    Om_f = G.T @ G + mu * np.eye(N)

    Om_s = (lam_r - lam_x) * G.T @ G
    F = G.T

    return (G, Th, Om_f, Om_s, F)

def calculate_Vdot(Om_s, r, F, c, noise=0):
    return F @ c + Om_s @ r + noise

def discrete_spikes(V, Th, Om_f, t, dt, timing, *args):
    if timing:
        cs = args[0]
        spktm = args[1]
    
    o = np.zeros_like(V)
    Vdiff = V - Th
    spiked = []
    spike_idx = np.argsort(Vdiff)
    while spike_idx.size > 0:
        n = spike_idx[-1]
        spike_idx = spike_idx[:-1]
        if V[n] > Th:
            if n in spiked:
                continue
            o[n] = 1
            if timing:
                tmp = int(cs[n])
                spktm[n,tmp] = t*dt
                cs[n] = tmp + 1

            V = V - Om_f[:,n]
            Vdiff = V - Th
            spike_idx = np.argsort(Vdiff)

    if timing:
        return (V, o, cs, spktm)

    return (V, o)

def simulate_ff(dt, S, N, J, G_scale, lam_x, lam_r, x, c, sig_noise, trials=10):
    G, Th, Om_f, Om_s, F = initialize_standard(N, J, G_scale, lam_x, lam_r, mu=5e-7)
    raster = True
    
    xh_all = []
    spktm_all = []
    o_all = []
    spike_counts = np.zeros((trials,N))
    for tr in range(trials):
        Vlast = np.zeros((N,))
        r = np.zeros((N,))
        o = np.zeros((N,S))
        xh = np.zeros_like(x)
        sigma_V = sig_noise * np.random.randn(N,S)
        cs = np.zeros((N,))
        spktm = np.zeros((N,S))

        for t in range(1,S):
            dV = -20*Vlast + F @ c[:,t-1] + Om_s @ r + sigma_V[:,t-1]
            V = Vlast + dV*dt

            V, ot, cs, spktm = discrete_spikes(V, Th, Om_f, t, dt, raster, cs, spktm)
            o[:,t] = ot
            r = (1 - lam_r*dt) * r + ot
            xh[:,t] = G @ r
            Vlast = V

        xh_all.append(xh)
        spktm_all.append(spktm)
        o_all.append(o)
        spike_counts[tr,:] = np.sum(o, axis=1)

    mn = np.mean(spike_counts, axis=0)
    std = np.std(spike_counts, axis=0)
    nz = np.nonzero(mn)

    ff = np.mean(std[nz] / mn[nz])

    return xh_all, spktm_all, o_all, ff

def simulate_balance(dt, S, N, J, G_scale, lam_x, lam_r, lam_s, x, c, mismatch_noise, trials=4):
    G, Th, Om_f, Om_s, F = initialize_standard(N, J, G_scale, lam_x, lam_r)
    Gpos = G.copy(); Gpos[Gpos<=0] = 0
    Gneg = -G.copy(); Gneg[Gneg<=0] = 0
    W_ei = -(Gpos.T@Gneg + Gneg.T@Gpos)
    W_ee = (Gpos.T@Gpos + Gneg.T@Gneg)
    raster = False
    
    xh_all = {}
    Ie_all = {}
    Ii_all = {}
    rmse_all = {}
    var_all = {}
    for mis in mismatch_noise:
        rmse_all[mis] = 0
        var_all[mis] = 0
        W_mis = mis*np.random.randn(N,N) + W_ei
        Om_s = W_ee + W_mis
        Om_f = W_ee + W_mis
        for tr in range(trials):
            Vlast = np.zeros((N,))
            Ie = np.zeros((N,S))
            Ii = np.zeros((N,S))
            o = np.zeros((N,))
            r = np.zeros((N,))
            xh = np.zeros_like(x)
            sigma_V = 1e-7 * np.random.randn(N,S)

            for t in range(1,S):
                ie = F@c[:,t-1] + (lam_r-lam_x)*W_ee@r - W_ee@(o/dt)
                Ie[:,t] = (1-lam_s*dt)*Ie[:,t-1] + dt * ie
                ii = (lam_r-lam_x)*W_mis@r - W_mis@(o/dt)
                Ii[:,t] = (1-lam_s*dt)*Ii[:,t-1] + dt * ii

                dV = F@c[:,t-1] + (lam_r-lam_x)*W_ee@r + (lam_r-lam_x)*W_mis@r + sigma_V[:,t-1]
                V = Vlast + dV*dt

                V, o = discrete_spikes(V, Th, Om_f, t, dt, raster)

                r = (1 - lam_r*dt) * r + o
                xh[:,t] = G @ r
                Vlast = V

            rmse = compute_rmse(x, xh)
            var = np.mean(np.std(Ie+Ii,axis=1)**2)
            rmse_all[mis] += rmse
            var_all[mis] += var

        Ie_all[mis] = Ie
        Ii_all[mis] = Ii
        xh_all[mis] = xh
        rmse_all[mis] /= trials
        var_all[mis] /= trials
        
    return xh_all, Ie_all, Ii_all, rmse_all, var_all

def simulate_noise(dt, S, N, J, G_scale, lam_x, lam_r, x, c, sig_noise, trials=4):
    
    G, Th, Om_f, Om_s, F = initialize_standard(N, J, G_scale, lam_x, lam_r, mu=5e-8)
    raster = True
    
    xh_all = {}
    spktm_all = {}
    rmse_all = {}
    for sn in sig_noise:
        rmse_all[sn] = 0
        for tr in range(trials):
            Vlast = np.zeros((N,))
            r = np.zeros((N,))
            xh = np.zeros_like(x)
            sigma_V = sn * np.random.randn(N,S)
            cs = np.zeros((N,))
            spktm = np.zeros((N,S))

            for t in range(1,S):
                dV = F @ c[:,t-1] + Om_s @ r + sigma_V[:,t-1]
                V = Vlast + dV*dt

                V, o, cs, spktm = discrete_spikes(V, Th, Om_f, t, dt, raster, cs, spktm)

                r = (1 - lam_r*dt) * r + o
                xh[:,t] = G @ r
                Vlast = V

            rmse = compute_rmse(x, xh)
            rmse_all[sn] += rmse

        xh_all[sn] = xh
        spktm_all[sn] = spktm
        rmse_all[sn] /= trials
        
    return xh_all, spktm_all, rmse_all

def simulate_ablation(dt, S, N, J, G_scale, lam_x, lam_r, x, c, rhos, trials=4):
    G, Th, Om_f, Om_s, F = initialize_standard(N, J, G_scale, lam_x, lam_r, mu=5e-8)
    raster = True
    
    xh_all = {}
    spktm_all = {}
    rmse_all = {}
    for p in rhos:
        rmse_all[p] = 0
        idx = np.random.choice(N, int(p*N))
        F[idx,:] = 0
        G[:,idx] = 0
        Om_s[idx,:] = 0; Om_s[:,idx] = 0
        Om_f[idx,:] = 0; Om_f[:,idx] = 0

        for tr in range(trials):
            Vlast = np.zeros((N,))
            r = np.zeros((N,))
            xh = np.zeros_like(x)
            sigma_V = 0 * np.random.randn(N,S)
            cs = np.zeros((N,))
            spktm = np.zeros((N,S))

            for t in range(1,S):
                dV = F @ c[:,t-1] + Om_s @ r + sigma_V[:,t-1]
                V = Vlast + dV*dt

                V, o, cs, spktm = discrete_spikes(V, Th, Om_f, t, dt, raster, cs, spktm)

                r = (1 - lam_r*dt) * r + o
                xh[:,t] = G @ r
                Vlast = V

            rmse = compute_rmse(x, xh)
            rmse_all[p] += rmse

        xh_all[p] = xh
        spktm_all[p] = spktm
        rmse_all[p] /= trials
        
    return xh_all, spktm_all, rmse_all

def simulate_delays(dt, ts, S, N, J, G_scale, lam_x, lam_r, x, c, delays, trials=4):
    G, Th, Om_f, Om_s, F = initialize_standard(N, J, G_scale, lam_x, lam_r, mu=5e-8)
    raster = True
    
    xh_all = {}
    spktm_all = {}
    rmse_all = {}
    for d in delays:
        rmse_all[d] = 0
        r = np.exp(-ts*lam_r)
        s = np.zeros_like(r)
        for t in range(1,S):
            s[t] = s[t-1] + dt*(-d*s[t-1] + r[t-1])
        scale = np.max(s)

        for tr in range(trials):
            Vlast = np.zeros((N,))
            Ilast = np.zeros((N,))
            r = np.zeros((N,))
            o = np.zeros((N,))
            xh = np.zeros_like(x)
            sigma_V = 1e-7 * np.random.randn(N,S)
            cs = np.zeros((N,))
            spktm = np.zeros((N,S))
            
            for t in range(1,S):
                reset = Om_f @ o
                idx = np.nonzero(o)
                if len(idx) > 0:
                    reset[idx] = 0
                
                dI = -d*Ilast + (F @ c[:,t-1] + Om_s @ r - reset + sigma_V[:,t-1])/scale
                I = Ilast + dt*dI
                V = Vlast + dt*I
                
                o = np.zeros((N,))
                n = np.argmax(V-Th)
                if V[n] > Th:
                    o[n] = 1/dt
                    V[n] -= Om_f[n,n]
                    r = (1-lam_r*dt) * r + dt*o
                    if raster:
                        tmp = int(cs[n])
                        spktm[n,tmp] = t*dt
                        cs[n] = tmp + 1
                else:
                    r = (1-lam_r*dt) * r
                
                xh[:,t] = G @ r
                Vlast = V
                Ilast = I
            
            rmse = compute_rmse(x, xh)
            rmse_all[d] += rmse

        xh_all[d] = xh
        spktm_all[d] = spktm
        rmse_all[d] /= trials
        
    return xh_all, spktm_all, rmse_all

def simulate_complete(dt, ts, S, N, J, G_scale, lam_x, lam_r, lam_s, x, c, noise_var, p, delay, mis, trials=5):
    mu = 5e-8
    G, Th, Om_f, Om_s, F = initialize_standard(N, J, G_scale, lam_x, lam_r, mu=mu)
    Wmis = mis*np.random.randn(N,N)

    Gpos = G.copy(); Gpos[Gpos<=0] = 0
    Gneg = -G.copy(); Gneg[Gneg<=0] = 0
    W_ei = -(Gpos.T@Gneg + Gneg.T@Gpos) + Wmis
    W_ee = (Gpos.T@Gpos + Gneg.T@Gneg)

    Om_f = W_ee + W_ei + mu*np.eye(N)
    Om_s = (lam_r - lam_x) * (W_ee + W_ei)

    raster = True
    spike_counts = np.zeros((trials,N))
    
    for tr in range(trials):
        r = np.exp(-ts*lam_r)
        s = np.zeros_like(r)
        for t in range(1,S):
            s[t] = s[t-1] + dt*(-delay*s[t-1] + r[t-1])
        scale = np.max(s)

        idx = np.random.choice(N, int(p*N))
        F[idx,:] = 0
        G[:,idx] = 0
        Om_s[idx,:] = 0; Om_s[:,idx] = 0
        Om_f[idx,:] = 0; Om_f[:,idx] = 0

        Vlast = np.zeros((N,))
        Ilast = np.zeros((N,))
        Ielast = np.zeros((N,))
        Iilast = np.zeros((N,))
        r = np.zeros((N,))
        o = np.zeros((N,S))
        Ie = np.zeros((N,S))
        Ii = np.zeros((N,S))
        ot = np.zeros((N,))
        xh = np.zeros_like(x)
        cs = np.zeros((N,))
        spktm = np.zeros((N,S))
        sigma_V = noise_var * np.random.randn(N,S)

        for t in range(1,S):
            reset_e = W_ee @ ot
            reset_i = W_ei @ ot
            idx = np.nonzero(ot)
            if len(idx) > 0:
                reset_e[idx] = 0
                reset_i[idx] = 0
            
            dIe = -delay*Ielast + (F@c[:,t-1] + (lam_r-lam_x)*W_ee@r - reset_e)/scale
            ie = Ielast + dt*dIe
            
            dIi = -delay*Iilast + ((lam_r-lam_x)*W_ei@r - reset_i)/scale
            ii = Iilast + dt*dIi
            
            Ie[:,t] = (1-lam_s*dt)*Ie[:,t-1] + dt * ie
            Ii[:,t] = (1-lam_s*dt)*Ii[:,t-1] + dt * ii
            
            reset = Om_f @ ot
            idx = np.nonzero(ot)
            if len(idx) > 0:
                reset[idx] = 0

            dI = -delay*Ilast + (F @ c[:,t-1] + Om_s @ r - reset + sigma_V[:,t-1])/scale
            I = Ilast + dt*dI
            V = Vlast + dt*I

            ot = np.zeros((N,))
            n = np.argmax(V-Th)
            if V[n] > Th:
                ot[n] = 1/dt
                V[n] -= Om_f[n,n]
                r = (1-lam_r*dt) * r + dt*ot
                if raster:
                    tmp = int(cs[n])
                    spktm[n,tmp] = t*dt
                    cs[n] = tmp + 1
            else:
                r = (1-lam_r*dt) * r

            o[:,t] = ot
            xh[:,t] = G @ r
            Vlast = V
            Ilast = I
            Ielast = ie
            Iilast = ii
        
        spike_counts[tr,:] = np.sum(o, axis=1)

    Inet = Ie + Ii
    mn = np.mean(spike_counts, axis=0)
    std = np.std(spike_counts, axis=0)
    nz = np.nonzero(mn)

    ff = np.mean(std[nz] / mn[nz])
    
    rmse = compute_rmse(x, xh)
    ei_var = np.mean(np.std(Inet, axis=1)**2)
    return xh, spktm, Inet, rmse, ff, ei_var
    