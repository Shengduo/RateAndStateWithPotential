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

# Generating a VT sequence for the spring slider
VTkwgs = {
        'logVDist' : True, 
        'Vrange' : [-2., 1.], 
        'Trange' : [0., 2.0], 
        'NofTpts' : 10, 
        'flag' : 'simple', 
        'nOfTerms' : 3, 
}

# myVT = GenerateVT(VTkwgs)
res = torch.load('./data/testSpringSlider0306_200.pth')
myVT = res['myVTs'][0]

# Generate a time sequence
# Spring slider parameters
kmg = [5000., 1., 9.8]

# Rate and state parameters
RSParams = [0.011, 0.016, 1. / 1.e-2, 0.58]

# Solver specific parameters
rtol, atol = 1.e-8, 1.e-10 
regularizedFlag = False

# Solver-specific settings
# solver = 'dopri5'
solver = 'implicit_adams'
max_iters = 20
# step_sizes = [pow(2., i) for i in np.linspace(-15, -11, num = 2)]
step_sizes = []

# Store the results
Frics = []
Vs = []
xs = []
legends = ["$\Delta t = {0}$ s".format(step_size) for step_size in step_sizes]

## Loop thru all step_sizes
for step_size in step_sizes:
    solver_options = {
        'max_iters' : max_iters, 
        'step_size' : step_size, 
    }

    # Initial condition
    y0 = torch.tensor([0., 1., 1. / RSParams[2]])
    myMFParams = MassFricParams(kmg, myVT.VT, RSParams, y0, lawFlag = "aging", regularizedFlag = regularizedFlag)

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
    print("=" * 25, " R & S friction ", "=" * 25)
    print("-"* 25, " Dt = {0}, {1} s.".format(step_size, time.time() - st), "-"* 25)

if len(Vs) > 0:
    res = {
        'VT' : myVT, 
        't' : myTimeSeq.t, 

    }

# Try loading the model
# Load the learnt model from storage
from FrictionNNModels import PotentialsFricCorrection, Loss, train1Epoch, PP, ReLUSquare, FricCorrection, load_model

modelPrefix = "Trial0216_combined_800"
# modelPrefix = "Trial0216_smallDRS_smallA_400"
dim_xi = 1

# Get correct device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

myModel = load_model(modelPrefix, device, dim_xi, True, NN_Flag=1)

