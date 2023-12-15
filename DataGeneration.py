import torch
import torchdiffeq
import pickle
import time
import torch.nn as nn
import scipy.optimize as opt
import numpy as np
from pathlib import Path
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from random import shuffle
from joblib import Parallel, delayed, effective_n_jobs
import json

# Function that generates VVs and tts
def genVVtt(totalNofSeqs, NofIntervalsRange, VVRange, VVLenRange, VVFirst, NofVVSteps = -1, DRS = 0.01):
    VVseeds = []
    VVseeds_len = []

    timesDRS = [2, 6]

    # Generate the seeds of VVs and tts
    for i in range(totalNofSeqs):
        NofSds = torch.randint(NofIntervalsRange[0], NofIntervalsRange[1], [1])

        # To make the log levels uniform distribution within the range
        VVseed = torch.rand([NofSds]) * (VVRange[1] - VVRange[0]) + VVRange[0]
        for j in range(1, len(VVseed)):
            if torch.abs(VVseed[j] - VVseed[j - 1]) <= 0.5:
                # print("shit!")
                VVseed[j] = VVseed[j - 1] + 0.5 * torch.sign(VVseed[j] - VVseed[j - 1])
        VVseed = torch.clip(VVseed, VVRange[0], VVRange[1])
        VVseed_len = (torch.rand([NofSds]) * (timesDRS[1] - timesDRS[0]) + timesDRS[0]) * DRS / torch.pow(10., VVseed) / 0.2
        VVseed_len = torch.ceil(VVseed_len).type(torch.int)
        # VVseed_len = 10 * torch.randint(VVLenRange[0], VVLenRange[1], [NofSds])
        while torch.sum(VVseed_len) < NofVVSteps:
            VVseed_len = torch.concat([VVseed_len, VVseed_len])
            VVseed = torch.concat([VVseed, VVseed])
        
        VVseed = VVseed[0:NofVVSteps]
        VVseed_len = VVseed_len[0:NofVVSteps]
        VVseed_len[-1] = NofVVSteps - torch.sum(VVseed_len[:-1])

        # if NofVVSteps > 0:
        #     VVseed_len = torch.floor(NofVVSteps / torch.sum(VVseed_len) * VVseed_len).type(torch.int)
        #     VVseed_len[-1] = NofVVSteps - torch.sum(VVseed_len[:-1])
        
        if VVFirst != -1:
            VVseed[0] = torch.log10(torch.tensor(VVFirst))

        VVseeds.append(VVseed)
        VVseeds_len.append(VVseed_len)
    
    print("VVFirst = ", VVFirst)

    # DEBUG LINES
    shit = [torch.sum(x) for x in VVseeds_len]
    print("Maximum VV length: ", max(shit), flush=True)
    print("Minimum VV length: ", min(shit), flush=True)
    
    VVs = []
    tts = []

    # Generate VVs and tts
    for idx, (VVseed, VVseed_len) in enumerate(zip(VVseeds, VVseeds_len)):
        VV = torch.zeros(torch.sum(VVseed_len))
        st = 0
        for j in range(len(VVseed_len)):
            VV[st : st + VVseed_len[j]] = torch.pow(10., VVseed[j])
            st += VVseed_len[j]
        VVs.append(VV)
        tt = torch.linspace(0., 0.2 * len(VV), len(VV))
        tts.append(tt)
    
    # data = {
    #     "VVs" : VVs, 
    #     "tts" : tts
    # }
    # torch.save(data, dataFilename)
    
    return VVs, tts

# Function to get ts, JumpIdxs (Jump indices for VVs), t_JumpIdxs (Jump indices for Vs), VtFuncs
def calVtFuncs(VVs, tts):
    # get JumpIdxs
    JumpIdxs = []
    for VV in VVs:
        JumpIdx = [0]
        for i in range(1, len(VV)):
            if VV[i] != VV[i - 1]:
                JumpIdx.append(i)
        if JumpIdx[-1] != len(VV) - 1:
            JumpIdx.append(len(VV) - 1)
        JumpIdxs.append(JumpIdx)

    # Get VtFuncs, ts, t_JumpIdxs
    VtFuncs = []
    ts = []
    t_JumpIdxs = []

    # Functions, ts and t_JumpIdxs
    t_tt_times = [10 for i in range(len(VVs))]
    for JumpIdx, VV, tt, t_tt_time in zip(JumpIdxs, VVs, tts, t_tt_times):
        VtFunc = []
        t = torch.linspace(tt[0], tt[-1], t_tt_time * len(tt))
        t_JumpIdx = [0]

        # DEBUG
        print("JumpIdx: ", JumpIdx)
        for i in range(len(JumpIdx) - 1):
            this_tt = tt[JumpIdx[i] : JumpIdx[i + 1] + 1].clone()
            this_VV = VV[JumpIdx[i] : JumpIdx[i + 1] + 1].clone()
            this_VV[-1] = this_VV[-2]
            VtFunc.append(interp1d(this_tt, this_VV))

            isIdx =  (t <= this_tt[-1])
            if isIdx[-1] == True:
                t_JumpIdx.append(len(isIdx))
            else:
                for j in range(len(isIdx)):
                    if isIdx[j] == False:
                        t_JumpIdx.append(j)
                        break
        
        t_JumpIdxs.append(t_JumpIdx)
        ts.append(t)
        VtFuncs.append(VtFunc)
    return ts, JumpIdxs, t_JumpIdxs, VtFuncs

# Divide the sequences and distribute to different workers
def cal_f_beta_parallel(beta, theta0, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True, 
                        n_workers = 16, pool = Parallel(n_jobs=16, backend='threading')):
    # For now, partition the tts such that each chunk only has one sequence
    t_splits = [[t] for t in ts]
    t_JumpIdx_splits = [[t_JumpIdx] for t_JumpIdx in t_JumpIdxs]
    tt_splits = [[tt] for tt in tts]
    JumpIdx_splits = [[JumpIdx] for JumpIdx in JumpIdxs]
    VtFunc_splits = [[VtFunc] for VtFunc in VtFuncs]

    # Get all the sequences
    res = pool(delayed(cal_f_beta)(
                beta, theta0, t_split, t_JumpIdx_split, tt_split, JumpIdx_split, VtFunc_split, std_noise, directCompute
            ) for t_split, t_JumpIdx_split, tt_split, JumpIdx_split, VtFunc_split in zip(t_splits, t_JumpIdx_splits, tt_splits, JumpIdx_splits, VtFunc_splits)
        )

    # Join the list
    Vs = [res[i][0] for i in range(len(res))]
    thetas = [res[i][1] for i in range(len(res))]
    fs = [res[i][2] for i in range(len(res))]
    
    Vs = [piece for this in Vs for piece in this] 
    thetas = [piece for this in thetas for piece in this] 
    fs = [piece for this in fs for piece in this] 

    # ## Debug lines
    # print("len(Vs): ", len(Vs))
    # print("len(thetas): ", len(thetas))
    # print("len(fs): ", len(fs))

    # print("Vs: ", Vs)
    # print("thetas: ", thetas)
    # print("fs: ", fs)
    # Partition the sets
    return Vs, thetas, fs

# Compute f history based on VtFunc and beta
def cal_f_beta(beta, theta0, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True):
    # theta0 = kwgs['theta0']

    # Get all sequences
    Vs = []
    thetas = []
    fs = []

    for t_this, t_JumpIdx, VtFunc, tt, JumpIdx in zip(ts, t_JumpIdxs, VtFuncs, tts, JumpIdxs):
        # V = torch.tensor(VtFunc(t), dtype=torch.float)
        V = torch.zeros(t_this.shape)
        theta = torch.zeros(t_this.shape)

        a = beta[0]
        b = beta[1]
        DRSInv = beta[2]
        fStar = beta[3]

        # Loop thru all sections of VtFunc
        theta0_this = theta0
        for index, vtfunc in enumerate(VtFunc):
            t_this_interval = t_this[t_JumpIdx[index] : t_JumpIdx[index + 1]] 

            # Append with the first and last tt
            t_this_interval = torch.cat([torch.tensor([tt[JumpIdx[index]]]), t_this_interval, torch.tensor([tt[JumpIdx[index + 1]]])])
            
            # Update V
            V[t_JumpIdx[index] : t_JumpIdx[index + 1]] = torch.tensor(vtfunc(t_this[t_JumpIdx[index] : t_JumpIdx[index + 1]]), dtype=torch.float)

            # Compute theta
            i = 0
            j = len(t_this_interval)
            if (t_this_interval[0] == t_this_interval[1]):
                i = i + 1
            if (t_this_interval[-1] == t_this_interval[-2]):
                j = -1

            if directCompute == True:
                V_thisStep = V[t_JumpIdx[index]]
                if V_thisStep * DRSInv < 1.e-4:
                    alp = V_thisStep * DRSInv
                    deltaT = t_this_interval[i : j] - t_this_interval[i]
                    theta_this = theta0_this + deltaT - alp * (deltaT * theta0_this + torch.square(deltaT) / 2.) \
                                 + alp * alp * theta0_this * torch.square(deltaT) / 2.
                else: 
                    InsideExp = -DRSInv * V_thisStep * (t_this_interval[i : j] - t_this_interval[i])
                    ExpTerm = torch.exp(InsideExp)
                    theta_this = (1 - (1 - DRSInv * V_thisStep * theta0_this) * ExpTerm) / (DRSInv * V_thisStep)
                # # DEBUG LINES
                # print("!"*100)
                # print("theta0_this: ", theta0_this)
                # print("theta_this[0]: ", theta_this[0])
                # print("DRSInv * V_thisStep: ", DRSInv * V_thisStep)
                # print("!"*100)

            else:
                thetaFunc = lambda t, theta: 1. - torch.tensor(vtfunc(torch.clip(t, tt[JumpIdx[index]], tt[JumpIdx[index + 1]])), dtype=torch.float) * theta * DRSInv
                theta_this = odeint(thetaFunc, theta0_this, t_this_interval[i : j], atol = 1.e-10, rtol = 1.e-8)
            

            # Update theta
            if i == 1:
                i = 0
            else:
                i = 1
            if j == -1:
                j = len(theta_this)
            else:
                j = -1

            theta[t_JumpIdx[index] : t_JumpIdx[index + 1]] = theta_this[i : j]
            theta0_this = theta_this[-1]

        # # Inside the cal_f_beta
        # print("="*10, " inside cal_f_beta ", "="*10)
        # print("V: ", V)
        # print("theta: ", theta)
        # print("="*10, " after cal_f_beta ", "="*10)
        f = fStar + a * torch.log(V / 1.e-6) + b * torch.log(1.e-6 * theta * DRSInv)
        mean_f = torch.mean(f);
        f = f + std_noise * mean_f * torch.randn(f.shape)
        Vs.append(V)
        thetas.append(theta)
        fs.append(f)
    
    # Vs = torch.stack(Vs)
    # thetas = torch.stack(thetas)
    # fs = torch.stack(fs)

    return Vs, thetas, fs

# Generate samples with hyperparameters kwgs
# Save kwgs and generated data to file
def generateSamples(kwgs):
    # Save hyper-parameters to json file
    prefix = kwgs["prefix"]
    jsonFile = "./data/" + prefix + ".json"
    with open(jsonFile, "w") as outfile:
        json.dump(kwgs, outfile)
    
    # Generate data
    VVs, tts = genVVtt(kwgs['totalNofSeqs'], kwgs['NofIntervalsRange'], kwgs['VVRange'], kwgs['VVLenRange'], 
                       1., #1. / kwgs['beta'][2] / kwgs['theta0'], 
                       kwgs["NofVVSteps"], 
                       DRS = 1. / kwgs['beta'][2])
    
    ts, JumpIdxs, t_JumpIdxs, VtFuncs = calVtFuncs(VVs, tts)
    Vs, thetas, fs = cal_f_beta_parallel(kwgs["beta"], kwgs['theta0'], ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.000, directCompute = True, 
                        n_workers = 16, pool = Parallel(n_jobs=16, backend='threading'))
    
    res = {
        "Vs" : torch.stack(Vs, dim=0), 
        "thetas" : torch.stack(thetas, dim=0), 
        "fs" : torch.stack(fs, dim=0), 
        "ts" : torch.stack(ts, dim=0), 
    }
    
    # Save the data file
    dataFileName = "./data/" + prefix + ".pt"
    torch.save(res, dataFileName)

# Smoother noisy inputs
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return torch.tensor(y_smooth, dtype=torch.float)

def generateVsAndts_Burigede(kwgs):
    Tmax = kwgs['Tmax']
    nTSteps = kwgs['nTSteps']
    N = kwgs['totalNofSeqs']
    Vrange = kwgs['VVRange']

    t = torch.linspace(0., Tmax, nTSteps)
    ts = torch.stack([t for i in range(N)], dim=0)
    
    Vs = []
    for i in range(N):
        # Number of intervals
        NofIntervals = torch.randint(kwgs['NofIntervalsRange'][0], kwgs['NofIntervalsRange'][1], [1])
        kRange = [(Vrange[1] - Vrange[0]) / (Tmax / NofIntervals), (Vrange[1] - Vrange[0]) / (Tmax / (1.5 * NofIntervals))]
        tChangePoints = Tmax * torch.sort(torch.rand(NofIntervals - 1))[0]
        a = torch.rand(1)
        if (a < 0.5):
            sign = 1.
        else:
            sign = -1
        
        # v = ((Vrange[1] - Vrange[0]) * torch.rand(1) + Vrange[0]) * torch.ones(nTSteps)
        v = torch.zeros(nTSteps)

        slope = (kRange[1] - kRange[0]) * torch.rand(nTSteps)
        sn = torch.ones(nTSteps)
        for tChangePoint in tChangePoints:
            sn[t >= tChangePoint] *= -1
        sn = sign * sn
        
        # sn = smooth(sn, 10)

        slope = slope * sn

        v[1:] = v[1:] + torch.cumulative_trapezoid(slope, t)
        v[v < Vrange[0]] = 2 * Vrange[0] - v[v < Vrange[0]]
        v[v > Vrange[1]] = 2 * Vrange[1] - v[v > Vrange[1]]
        # v = smooth(v, 20)
        v = savitzky_golay(v, window_size=11, order=3, deriv=0, rate=1)
        v = torch.pow(10., v)
        Vs.append(v)
    
    Vs = torch.stack(Vs, dim = 0)

    return Vs, ts


        

# Generate samples with hyperparameters kwgs
# Save kwgs and generated data to file, using burigede's generation
def generateSamples_Burigede(kwgs):
    # Save hyper-parameters to json file
    prefix = kwgs["prefix"]
    jsonFile = "./data/" + prefix + ".json"
    with open(jsonFile, "w") as outfile:
        json.dump(kwgs, outfile)
    
    # Generate data
    Vs, ts = generateVsAndts_Burigede(kwgs)
    
    beta = kwgs['beta']
    thetas = []
    
    for i, (V, t) in enumerate(zip(Vs, ts)):
        VTfunc = interp1d(t, V)
        thetaFunc = lambda tau, theta: 1. - torch.tensor(VTfunc(torch.clip(tau, t[0], t[-1])), dtype=torch.float) * theta * beta[2]
        theta_this = odeint(thetaFunc, torch.tensor(kwgs['theta0']), t, atol = 1.e-10, rtol = 1.e-8)
        thetas.append(theta_this)

    thetas = torch.stack(thetas, dim=0)
    fs = beta[3] + beta[0] * torch.log(Vs / 1.e-6) + beta[1] * torch.log(1.e-6 * thetas * beta[2])

    res = {
        "Vs" : Vs, 
        "thetas" : thetas, 
        "fs" : fs, 
        "ts" : ts, 
    }
    
    # Save the data file
    dataFileName = "./data/" + prefix + ".pt"
    torch.save(res, dataFileName)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    y = y.numpy()
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    res = np.convolve( m[::-1], y, mode='valid')
    return torch.tensor(res, dtype=torch.float)