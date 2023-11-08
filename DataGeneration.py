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
def genVVtt(totalNofSeqs, NofIntervalsRange, VVRange, VVLenRange, VVFirst, NofVVSteps = -1):
    VVseeds = []
    VVseeds_len = []

    # Generate the seeds of VVs and tts
    for i in range(totalNofSeqs):
        NofSds = torch.randint(NofIntervalsRange[0], NofIntervalsRange[1], [1])
        VVseed = torch.randint(VVRange[0], VVRange[1], [NofSds])
        VVseed_len = 10 * torch.randint(VVLenRange[0], VVLenRange[1], [NofSds])
        if NofVVSteps > 0:
            VVseed_len = torch.floor(NofVVSteps / torch.sum(VVseed_len) * VVseed_len).type(torch.int)
            VVseed_len[-1] = NofVVSteps - torch.sum(VVseed_len[:-1])
        
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
                       -1, #1. / kwgs['beta'][2] / kwgs['theta0'], 
                       kwgs["NofVVSteps"])
    ts, JumpIdxs, t_JumpIdxs, VtFuncs = calVtFuncs(VVs, tts)
    Vs, thetas, fs = cal_f_beta_parallel(kwgs["beta"], kwgs['theta0'], ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True, 
                        n_workers = 16, pool = Parallel(n_jobs=16, backend='threading'))
    
    res = {
        "Vs" : Vs, 
        "thetas" : thetas, 
        "fs" : fs, 
        "ts" : ts, 
    }
    
    # Save the data file
    dataFileName = "./data/" + prefix + ".pt"
    torch.save(res, dataFileName)