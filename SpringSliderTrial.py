# Import classes for generating sequences
import torch
from torch import nn
import numpy as np
from GenerateVT import GenerateVT
from MassFricParams import MassFricParams
from TimeSequenceGen import TimeSequenceGen
from TimeSequenceGen_NN import TimeSequenceGen_NN
from matplotlib import pyplot as plt
from matplotlib import rc
from FrictionNNModels import plotGenVXFric
import time
from pathlib import Path

# Generating a VT sequence for the spring slider
VTkwgs = {
        'logVDist' : True, 
        'Vrange' : [-2., 1.], 
        'Trange' : [0., 2.0], 
        'NofTpts' : 10, 
        'flag' : 'simple', 
        'nOfTerms' : 3, 
}

# Set Path
PATH = "./data/RSvsNNSpringSliderAcc0522_halves"
Path(PATH).mkdir(parents=True, exist_ok=True)

## Load trained NN model
from FrictionNNModels import PotentialsFricCorrection, Loss, train1Epoch, PP, ReLUSquare, FricCorrection, load_model

modelPrefix = "Trial_0216_0521SS_400"
# modelPrefix = "Trial0216_smallDRS_smallA_400"
dim_xi = 1

# Get correct device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def memory_stats():
    print("Memory allocated: ", torch.cuda.memory_allocated()/1024**2)
    print("Memory cached: ", torch.cuda.memory_reserved()/1024**2)

# Release GPU memory
import gc
gc.collect()
torch.cuda.empty_cache()
memory_stats()

# device = torch.device("cpu")
# Load NN model
myModel = load_model(modelPrefix, device, dim_xi, True, NN_Flag=1)

## Load dataset of VT sequences
# myVT = GenerateVT(VTkwgs)
data = torch.load('./data/testSpringSlider0522_200.pt')
step_sizes = [pow(2., i) for i in np.linspace(-13.5, -11.5, num = 3)]

# Loop thru all sequences
for i in range(0, len(data['myVTs'])):
    print("=" * 30, " Seq {0} ".format(i), "=" * 30, flush=True)
    myVT = data['myVTs'][i]

    # Generate a time sequence
    # Spring slider parameters
    kmg = [5000., 1., 9.8]
    kmg[0] = data['ks'][i]

    # Rate and state parameters
    RSParams = [0.011, 0.016, 1. / 1.e-2, 0.58]

    # Solver specific parameters
    rtol, atol = 1.e-8, 1.e-10 
    regularizedFlag = True

    # Solver-specific settings
    # solver = 'dopri5'
    solver = 'implicit_adams'
    max_iters = 20

    # Store the results
    Frics = []
    Vs = []
    xs = []
    legends = ["$\Delta t = {0}$ s".format(step_size) for step_size in step_sizes]

    # Initial condition
    y0 = torch.tensor([0., 1., 1. / RSParams[2]])
    myMFParams = MassFricParams(kmg, myVT.VT, RSParams, y0, lawFlag = "aging", regularizedFlag = regularizedFlag)

    # Print out RS solve
    print("*" * 25, " RS solve implicit ", "*" * 25, flush=True)

    ## Loop thru all step_sizes
    for step_size in step_sizes:
        solver_options = {
            'max_iters' : max_iters, 
            'step_size' : step_size, 
        }

        # Start timer
        st = time.time()

        # Set for my sequence
        myTimeSeq = TimeSequenceGen(myVT.Trange[1], 
                                    10, 
                                    myMFParams, 
                                    rtol, 
                                    atol, 
                                    regularizedFlag, 
                                    solver, 
                                    solver_options)
        
        Vs.append(myTimeSeq.default_y[1, :])
        xs.append(myTimeSeq.default_y[0, :])
        Frics.append(myTimeSeq.Fric)
        # print("=" * 25, " R & S friction ", "=" * 25)
        print("-"* 25, " Dt = {0}, {1} s.".format(step_size, time.time() - st), "-"* 25)

    if len(Vs) > 0:
        res = {
            'kmg' : kmg, 
            'VT' : myVT, 
            't' : myTimeSeq.t, 
            'Vs' : Vs, 
            'xs' : xs, 
            'Frics' : Frics, 
            'step_sizes' : step_sizes, 
        }
        torch.save(res, PATH + "/res_RS_im_reg_seq_{0}.pth".format(i))

    # Print out RS solve
    Frics = []
    Vs = []
    xs = []
    print("*" * 25, " RS solve explicit ", "*" * 25, flush=True)
    solver = 'rk4'

    ## Loop thru all step_sizes
    for step_size in step_sizes:
        solver_options = {
            'max_iters' : max_iters, 
            'step_size' : step_size, 
        }

        # Start timer
        st = time.time()

        # Set for my sequence
        myTimeSeq = TimeSequenceGen(myVT.Trange[1], 
                                    10, 
                                    myMFParams, 
                                    rtol, 
                                    atol, 
                                    regularizedFlag, 
                                    solver, 
                                    solver_options)
        
        Vs.append(myTimeSeq.default_y[1, :])
        xs.append(myTimeSeq.default_y[0, :])
        Frics.append(myTimeSeq.Fric)
        # print("=" * 25, " R & S friction ", "=" * 25)
        print("-"* 25, " Dt = {0}, {1} s.".format(step_size, time.time() - st), "-"* 25)

    if len(Vs) > 0:
        res = {
            'kmg' : kmg, 
            'VT' : myVT, 
            't' : myTimeSeq.t, 
            'Vs' : Vs, 
            'xs' : xs, 
            'Frics' : Frics, 
            'step_sizes' : step_sizes, 
        }
        torch.save(res, PATH + "/res_RS_ex_reg_seq_{0}.pth".format(i))

    # NN explicit solve
    solver = 'rk4'
    max_iters = 20

    step_size_NNs = [step_size for step_size in step_sizes]
    # step_size_NNs = []
    # step_size_NNs = [pow(2., i) for i in np.linspace(-9, -5, num = 3)]

    # Store the results
    Fric_NNs = []
    V_NNs = []
    x_NNs = []
    legend_NNs = ["$\Delta t = {0}$ s".format(step_size) for step_size in step_sizes]

    print("*" * 25, " NN explicit solve ", "*" * 25, flush=True)
    for step_size in step_size_NNs:
        solver_options = {
            'max_iters' : max_iters, 
            'step_size' : step_size, 
        }

        # Start the timer
        st = time.time()
        
        # Compute slip rate for the same input sequence using NN model
        myTimeSeqNN = TimeSequenceGen_NN(myVT.Trange[1], 
                                        10, 
                                        myMFParams, 
                                        myModel, 
                                        rtol, 
                                        atol, 
                                        solver, 
                                        solver_options,  
                                        fOffSet = 0.5109, 
                                        scaling_factor = 50.)

        V_NNs.append(myTimeSeqNN.default_y[1, :])
        x_NNs.append(myTimeSeqNN.default_y[0, :])
        Fric_NNs.append(myTimeSeqNN.Fric)
        gc.collect()
        torch.cuda.empty_cache()
        print("-"* 25, " Dt = {0}, {1} s.".format(step_size, time.time() - st), "-"* 25, flush=True)

    if len(step_size_NNs) > 0:
        res = {
            'kmg' : kmg, 
            'VT' : myVT, 
            't' : myTimeSeqNN.t, 
            'Vs' : V_NNs, 
            'xs' : x_NNs, 
            'Frics' : Fric_NNs, 
            'step_sizes' : step_size_NNs, 
            'legend_NNs' : legend_NNs, 
        }
        torch.save(res, PATH + "/res_NN_ex_seq_{0}.pth".format(i))
    
    # NN Implicit solve
    print("*" * 25, " NN implicit solve ", "*" * 25, flush=True)
    Fric_NNs = []
    V_NNs = []
    x_NNs = []
    solver = 'implicit_adams'

    for step_size in step_size_NNs:
        solver_options = {
            'max_iters' : max_iters, 
            'step_size' : step_size, 
        }

        # Start the timer
        st = time.time()
        
        # Compute slip rate for the same input sequence using NN model
        myTimeSeqNN = TimeSequenceGen_NN(myVT.Trange[1], 
                                        10, 
                                        myMFParams, 
                                        myModel, 
                                        rtol, 
                                        atol, 
                                        solver, 
                                        solver_options,  
                                        fOffSet = 0.5109, 
                                        scaling_factor = 50.)

        V_NNs.append(myTimeSeqNN.default_y[1, :])
        x_NNs.append(myTimeSeqNN.default_y[0, :])
        Fric_NNs.append(myTimeSeqNN.Fric)
        gc.collect()
        torch.cuda.empty_cache()
        print("-"* 25, " Dt = {0}, {1} s.".format(step_size, time.time() - st), "-"* 25, flush=True)

    if len(step_size_NNs) > 0:
        res = {
            'kmg' : kmg, 
            'VT' : myVT, 
            't' : myTimeSeqNN.t, 
            'Vs' : V_NNs, 
            'xs' : x_NNs, 
            'Frics' : Fric_NNs, 
            'step_sizes' : step_size_NNs, 
            'legend_NNs' : legend_NNs, 
        }
        torch.save(res, PATH + "/res_NN_im_seq_{0}.pth".format(i))

