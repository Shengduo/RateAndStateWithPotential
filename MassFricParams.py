# Import packages
import torch
import torch.nn as nn
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import scipy.optimize as opt
import pickle
import numpy as np
import time


# torch.set_default_dtype(torch.float)

"""
Class MassFricParams, manages data of a mass block sliding on rate-and-state friction surface, contains 
    Data:
        k : Spring stiffness
        m : Mass of the block
        V : Leading head speed of the spring
        g : Gravity
        RSParams : rate and state parameters, torch.tensor([a, b, DRS, f*])
        y0 : torch.tensor([initial x_1, initial v_1, initial state variable])
"""
class MassFricParams: 
    # Constructor
    def __init__(self, kmg, VT, RSParams, y0, lawFlag = "aging", regularizedFlag = True):
        # Define constant parameters k, m and g
        self.k = kmg[0]
        self.m = kmg[1]
        self.g = kmg[2]
        
        # Get the VT relation
        # print("VT: ", VT)
        self.V = VT[0, :]
        self.T = VT[1, :]
        
        # Get the displacement at T
        self.S = torch.zeros(self.V.shape)
        self.S[1:] = torch.cumulative_trapezoid(self.V, self.T)
        
        self.RSParams = RSParams
        self.y0 = y0
        # self.y0[1] = VT[0, 0]

        self.lawFlag = lawFlag
        self.regularizedFlag = regularizedFlag
        
        # Get the function of V, S at T
        self.vtFunc = interp1d(self.T, self.V)
        self.stFunc = interp1d(self.T, self.S)
        
    # Define the function that gives V at t
    def VatT_interp(self, t):
        # Piecewise linear interplation
        if t < self.T[0]:
            v = self.V[0]
        elif t > self.T[-1]:
            v = self.V[-1]
        else:
            v = self.vtFunc(t)
        return torch.tensor(v, dtype=torch.float)
    
    # Define the function that gives S at t
    def SatT_interp(self, t):
        # Piecewise linear interplation
        if t < self.T[0]:
            s = self.S[0]
        elif t > self.T[-1]:
            s = self.S[-1]
        else:
            s = torch.from_numpy(self.stFunc(t))
        return s
    
    # Output the information of this class
    def print_info(self):
        print("-" * 20, " Mass and spring parameters ", "-"*20)
        print('k:        ', self.k)
        print('m:        ', self.m)
        print('g:        ', self.g)
        print('\n')
        
        print("-" * 20, " Rate-and-state parameters ", "-"*20)
        print('fr:       ', self.RSParams[3])
        print('a:        ', self.RSParams[0])
        print('b:        ', self.RSParams[1])
        # print('DRS:      ', self.RSParams[2])
        print('1 / DRS:  ', self.RSParams[2])
        print('y0:       ', self.y0)
        print('law:      ', self.lawFlag)
        # # Plot V at t
        # plt.figure()
        # plt.plot(self.T, self.V, linewidth = 2.0)
        # plt.show()
        