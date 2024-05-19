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
class TimeSequenceGen:
    # Constructor
    def __init__(self, T, NofTPts, MFParams, rtol = 1.e-6, atol = 1.e-8, regularizedFlag = True, solver = 'dopri5', solver_options = dict()):
        # print("shit.")
        # Load the parameters
        self.T = T
        self.t = torch.linspace(0., T, NofTPts * len(MFParams.T))
        self.MFParams = MFParams
        self.rtol = rtol
        self.atol = atol
        self.regularizedFlag = regularizedFlag
        self.solver = solver
        self.solver_options = solver_options

        # Generate the sequence
        st = time.time()
        
        # Adjust the initial x so that the sequence starts with steady state
        self.DyDt0 = self.DyDt(torch.tensor(0.), self.MFParams.y0);
        self.y0 = self.MFParams.y0
        self.y0[0] += self.MFParams.m / self.MFParams.k * self.DyDt0[1]

        self.default_y = self.calculateYAtT(self.t)
        self.time_cost = time.time() - st
        # print("Time cost to generate the sequence: ", self.time_cost)

        
        
    # Function DyDt, DyDt = f(t, y)
    def DyDt(self, t, y):
        
        # Need to use regularized rate-and-state
        a = self.MFParams.RSParams[0]
        b = self.MFParams.RSParams[1]
        # DRS = self.MFParams.RSParams[2]
        DRSInv = self.MFParams.RSParams[2]
        fStar = self.MFParams.RSParams[3]
        
        # Regularized rate-and-state friction
#         tau = self.N * a * torch.asinh(
#                    self.Ds / 2. / Vr * torch.exp((fr + b * torch.log(Vr * self.theta / DRS)) / a)
#                    )
        # Get the displacement at t of the spring
        
        if self.regularizedFlag:
            # DyDt = torch.tensor([y[1], 
            #                      self.MFParams.k / self.MFParams.m * (self.MFParams.SatT_interp(t) - y[0]) - \
            #                      self.MFParams.g * (a * torch.asinh(
            #                          y[1] / 2.e-6 * torch.exp((fStar + b * torch.log(1.e-6 * y[2] / DRS)) / a)
            #                      )), 
            #                      # 1 - 1.e-6 * y[2] / DRS]) 
            #                      1 - y[2] * y[1] / DRS])
            DyDt = torch.tensor([y[1], 
                                 self.MFParams.k / self.MFParams.m * (self.MFParams.SatT_interp(t) - y[0]) - \
                                 self.MFParams.g * (a * torch.asinh(
                                     y[1] / 2.e-6 * torch.exp((fStar + b * torch.log(1.e-6 * y[2] * DRSInv)) / a)
                                 )), 
                                 # 1 - 1.e-6 * y[2] / DRS]) 
                                 1 - y[2] * y[1] * DRSInv])
        else:
            # DyDt = torch.tensor([y[1], 
            #                      self.MFParams.k / self.MFParams.m * (self.MFParams.SatT_interp(t) - y[0]) - \
            #                      self.MFParams.g * (fStar + \
            #                                         a * torch.log(y[1] / 1.e-6) + \
            #                                         b * torch.log(1.e-6 * y[2] / DRS)), 
            #                      # 1 - 1.e-6 * y[2] / DRS]) 
            #                      1 - y[2] * y[1] / DRS])     
            DyDt = torch.tensor([y[1], 
                                 self.MFParams.k / self.MFParams.m * (self.MFParams.SatT_interp(t) - y[0]) - \
                                 self.MFParams.g * (fStar + \
                                                    a * torch.log(y[1] / 1.e-6) + \
                                                    b * torch.log(1.e-6 * y[2] * DRSInv)), 
                                 # 1 - 1.e-6 * y[2] / DRS]) 
                                 1 - y[2] * y[1] * DRSInv])

        # Check if slip law should be used
        if self.MFParams.lawFlag == "slip":
            DyDt[2] = -y[2] * y[1] * DRSInv * torch.log(y[2] * y[1] * DRSInv)

        # DEBUG LINES
#         print("-" * 30)
#         print('t = ', t)
#         print('y = ', y)
#         print('DyDt = ', DyDt)
        
        return DyDt
    
    
    # Generate the sequence of y(t) = [x_1(t), v_1(t), theta(t)]
    def calculateYAtT(self, t):
        y = odeint(self.DyDt, self.y0, t, 
                   rtol = self.rtol, atol = self.atol, method = self.solver, options = self.solver_options)
        
        y = torch.transpose(y, 0, 1)

        # Calculate friction force also
        a = self.MFParams.RSParams[0]
        b = self.MFParams.RSParams[1]
        # DRS = self.MFParams.RSParams[2]
        DRSInv = self.MFParams.RSParams[2]
        fStar = self.MFParams.RSParams[3]

        self.Fric = fStar + a * torch.log(y[1, :] / 1.e-6) + b * torch.log(DRSInv * y[2, :] * 1.e-6)

        return y
    
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
        axs[1][0].semilogy(t, y[2, :], linewidth=3.0)
        axs[1][0].semilogy(t, 1. / self.MFParams.RSParams[2] / y[1, :], linewidth=2.0)
        axs[1][0].set_xlabel('Time [s]', fontsize=20)
        axs[1][0].set_ylabel('State Variable $\\theta(t)\ \mathrm{[s]}$', fontsize=20)
        axs[1][0].legend(['True', 'Steady state'], loc='best', fontsize=20)
        axs[1][0].grid()

        # Plot friction coefficient(t)
        axs[1][1].tick_params(axis='both', which='major', labelsize=20)
        axs[1][1].plot(t, self.Fric, linewidth=3.0)
        axs[1][1].set_xlabel('Time [s]', fontsize=20)
        axs[1][1].set_ylabel('Fric Coeff.', fontsize=20)
        axs[1][1].grid()
