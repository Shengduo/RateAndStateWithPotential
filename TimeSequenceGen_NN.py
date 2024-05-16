## Import standard librarys
import torch
import torchdiffeq
import pickle
import time
import torch.nn as nn
import scipy.optimize as opt
import numpy as np

from torchdiffeq import odeint
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib import rc
from MassFricParams import MassFricParams

"""
Class TimeSequenceGen, container for a Generated time sequence containing 
    Data:
        MFParams: Mass and friction parameters for the system
        T: Length of calculation
        
    Method:
        __init__ : Constructor
        calculateYAtT: Generate the sequence of [x_1, v_1, theta]
        
"""
class TimeSequenceGen_NN:
    # Constructor
    def __init__(self, T, NofTPts, MFParams, NNModel, rtol = 1.e-6, atol = 1.e-8, solver = 'dopri5', solver_options = dict(), 
                 fOffSet = 0.5109, scaling_factor = 50.):
        # Load the parameters
        self.T = T
        self.t = torch.linspace(0., T, NofTPts * len(MFParams.T))
        self.MFParams = MFParams
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.solver_options = solver_options
        self.NNModel = NNModel 
        self.fOffSet = fOffSet
        self.scaling_factor = scaling_factor
        
        # Generate the sequence
        st = time.time()
        self.default_y = self.calculateYAtT(self.t)
        self.time_cost = time.time() - st
        # print("Time cost to generate the sequence: ", self.time_cost)
        
    # Function DyDt, DyDt = f(t, y)
    def DyDt(self, t, y):
        # Calculate friction and xiDot
        Fric, xiDot = self.calFric(t, y)

        # Initialize DyDt
        DyDt = torch.zeros(2 + self.NNModel.dim_xi)
        DyDt[0] = y[1]
        DyDt[1] = self.MFParams.k / self.MFParams.m * (self.MFParams.SatT_interp(t.detach()) - y[0]) - \
                  self.MFParams.g * Fric 
        DyDt[2:] = xiDot 
        return DyDt
    
    # Calculate friction at (t, y)
    def calFric(self, t, y):
        # First compute friction force
        W_in = y[0].clone().reshape([1]).requires_grad_()
        D_dagger_in = y[1:].clone().requires_grad_()
        
        # Send to GPU
        W_in = W_in.to(self.NNModel.device)
        D_dagger_in = D_dagger_in.to(self.NNModel.device)

        W = self.NNModel.W(W_in)
        D_dagger = self.NNModel.D_dagger(D_dagger_in)

        # Take the derivatives
        dDaggerDDDaggerIn = torch.autograd.grad(outputs=D_dagger, inputs=D_dagger_in, create_graph=True)[0]
        dWDWIn = torch.autograd.grad(outputs=W, inputs=W_in, create_graph=True)[0]
        Fric = dWDWIn + dDaggerDDDaggerIn[0]
        D_in = -dDaggerDDDaggerIn[1:].clone().requires_grad_()
        D = self.NNModel.D(D_in)
        xiDot = torch.autograd.grad(outputs=D, inputs=D_in, create_graph=True)[0]

        
        # Send back Fric and xiDot. 
        Fric = Fric.to(torch.device("cpu"))
        xiDot = xiDot.to(torch.device("cpu"))
        
        # Re-scale friction
        Fric = (Fric - self.fOffSet) / self.scaling_factor + self.fOffSet
        
        # Delete gpu variables
        del W_in, D_dagger_in, W, D_dagger, dDaggerDDDaggerIn, dWDWIn, D_in, D 

        # Return friction and time derivative of xi
        return Fric, xiDot
    
    # Generate the sequence of y(t) = [x_1(t), v_1(t), theta(t)]
    def calculateYAtT(self, t):
        y0In = torch.zeros(2 + self.NNModel.dim_xi)
        y0In[0:2] = self.MFParams.y0[0:2]
        y = odeint(self.DyDt, y0In, t, 
                   rtol = self.rtol, atol = self.atol, method = self.solver, options = self.solver_options)
        y = torch.transpose(y, 0, 1)

        # Calculate friction force
        self.Fric = torch.zeros(y.shape[1])
        for idx in range(len(t)):
            self.Fric[idx], _ = self.calFric(t[idx], y[:, idx])
        self.Fric = self.Fric.detach()

        return y.detach()
    
    # Visualize the sequence of y
    def plotY(self, t, y):
        # Plot Sequence of V(t) and N(t) given sample-index
        f, axs = plt.subplots(2, 2, figsize = (15, 15))

        # Plot x_1(t)
        axs[0][0].tick_params(axis='both', which='major', labelsize=20)
        axs[0][0].plot(t, y[0, :], linewidth=2.0)
        axs[0][0].set_xlabel('Time [s]', fontsize=20)
        axs[0][0].set_ylabel('Slip $x_1(t)\  \mathrm{[m]}$', fontsize=20)
        # axs[0][0].set_ylim([1e-15, 1e2])
        axs[0][0].grid()

        # Plot v_1(t)
        axs[0][1].tick_params(axis='both', which='major', labelsize=20)
        axs[0][1].semilogy(t, y[1, :], linewidth=2.0)
        axs[0][1].set_xlabel('Time [s]', fontsize=20)
        axs[0][1].set_ylabel('Slip rate $v_1(t)\ \mathrm{[m/s]}$', fontsize=20)
        # axs[0][1].set_ylim([0, 15])
        axs[0][1].grid()

        # Plot theta(t)
        axs[1][0].tick_params(axis='both', which='major', labelsize=20)
        # axs[1][0].semilogy(t, y[2, :], linewidth=3.0)
        # axs[1][0].semilogy(t, 1. / self.MFParams.RSParams[2] / y[1, :], linewidth=2.0)
        # axs[1][0].set_xlabel('Time [s]', fontsize=20)
        # axs[1][0].set_ylabel('State Variable $\\theta(t)\ \mathrm{[s]}$', fontsize=20)
        # axs[1][0].legend(['True', 'Steady state'], loc='best', fontsize=20)
        # axs[1][0].grid()

        # Plot friction coefficient(t)
        axs[1][1].tick_params(axis='both', which='major', labelsize=20)
        axs[1][1].tick_params(axis='both', which='major', labelsize=20)
        axs[1][1].plot(t, self.Fric, linewidth=3.0)
        axs[1][1].set_xlabel('Time [s]', fontsize=20)
        axs[1][1].set_ylabel('Fric Coeff.', fontsize=20)
        axs[1][1].grid()
