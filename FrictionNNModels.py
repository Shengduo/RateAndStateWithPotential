#Ｉmport necessary packages
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import xitorch
from xitorch.optimize import rootfinder
import optuna
from torch.utils.data import TensorDataset, DataLoader
import joblib 
from pathlib import Path

# Memory management on GPU
import gc

# Import time
import time


# Different Potentials with D correction
import torch.optim as optim
# Define MLP for potentials
class ReLUSquare(nn.Module): 
    def __init__(self): 
        super(ReLUSquare, self).__init__() 
        self.fc = nn.ELU()
  
    def forward(self, x): 
        return torch.pow(self.fc(x), 1)
    
class PP(nn.Module):
    # Constructor
    def __init__(self, NNs, input_dim = 1, output_dim = 1):
        super().__init__()
        self.activation = ReLUSquare()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, NNs[0]), 
            # nn.BatchNorm1d(num_features=NNs[0]), 
            self.activation,
        )
        
        for i in range(len(NNs) - 1):
            self.fc.append(nn.Linear(NNs[i], NNs[i + 1]))
            # self.fc.append(nn.BatchNorm1d(num_features=NNs[i + 1]))
            self.fc.append(self.activation)

        
        self.fc.append(nn.Linear(NNs[-1], output_dim))
        # self.fc.append(nn.BatchNorm1d(NNs[-1]))
        # self.fc.append(self.activation)
    
    # Forward function
    def forward(self, x):
        # print("x.shape in PP: ", x.shape, flush=True)
        return self.fc(x)


class PotentialsFricCorrection:
    # Initialization of W and D
    def __init__(self, kwgsPot):
        self.dim_xi = kwgsPot["dim_xi"]
        self.NNs_W = kwgsPot["NNs_W"]
        self.NNs_D = kwgsPot["NNs_D"]
        self.NNs_D_dagger = kwgsPot["NNs_D_dagger"]
        self.W = PP(self.NNs_W, input_dim = 1, output_dim = 1)
        self.D = PP(self.NNs_D, input_dim = self.dim_xi, output_dim = 1)
        self.D_dagger = PP(self.NNs_D_dagger, input_dim = 1 + self.dim_xi, output_dim = 1)
        self.optim_W = optim.Adam(self.W.parameters(), lr=kwgsPot["learning_rate"])
        self.optim_D = optim.Adam(self.D.parameters(), lr=kwgsPot["learning_rate_D"])
        self.optim_D_dagger = optim.Adam(self.D_dagger.parameters(), lr=kwgsPot["learning_rate_D_dagger"])
        
        # Device
        self.device = kwgsPot["device"]
        
        # Multi-GPU data parallel
        self.W = nn.DataParallel(self.W)
        self.D = nn.DataParallel(self.D)
        self.D_dagger = nn.DataParallel(self.D_dagger)
        
        self.W.to(self.device)
        self.D.to(self.device)
        self.D_dagger.to(self.device)
        
    # Calculate f 
    def calf(self, x, xDot, t):
        # Initialize Vs
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        # xis[:, :, :] = 1. 
        
        
        # Loop through time steps
        
        if self.dim_xi > 0:
            xiNext = torch.zeros([batch_size, self.dim_xi], requires_grad=True, device=self.device)
            # xi0 = torch.zeros([batch_size, self.dim_xi], requires_grad=True, device=self.device)
            
            # List of fs
            list_fs = []
            # list_xis = [xi0]
            
            for idx in range(x.shape[1]):
                # Load xi_curr
                xiCurr = xiNext

                # f = \partial W / \partial V
                X_D_dagger = torch.concat([xDot[:, idx:idx + 1], xiCurr], dim = 1).requires_grad_()
                # X_W.to(self.device)
                D_dagger = torch.sum(self.D_dagger(X_D_dagger))

                this_piece = torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0]

                # Solve for \dot{\xi} + \partial W / \partial \xi = 0
                dD_daggerdXi = this_piece[:, 1:]
                dD_daggerdXDot = this_piece[:, 0:1]

                X_W = x[:, idx:idx+1].requires_grad_()
                W = torch.sum(self.W(X_W))
                dWdX = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0]

                list_fs.append(dD_daggerdXDot + dWdX.reshape([-1, 1]))

                # XiDot = dD^*/d\dot{d} (-dD^\dagger / dXi)
                if idx < x.shape[1] - 1:
                    this_input = -dD_daggerdXi.clone().requires_grad_()
                    D = torch.sum(self.D(this_input))
                    xiNext = xiCurr + torch.autograd.grad(outputs=D, inputs=this_input, create_graph=True)[0] * (t[:, idx + 1:idx + 2] - t[:, idx:idx + 1])
                    # list_xis.append(xiNext)
                    
                    del this_input, dD_daggerdXi, dD_daggerdXDot, W, X_W, dWdX, D, this_piece, X_D_dagger, D_dagger 
                else:
                    del dD_daggerdXi, dD_daggerdXDot, W, X_W, dWdX, this_piece, X_D_dagger, D_dagger
                    
            self.fs = torch.concat(list_fs, dim=1)
            del xiNext, xiCurr
            
        else:
            X_W = x.clone().reshape([x.shape[0], x.shape[1], 1]).requires_grad_()
            # print(X_W)
            W = torch.sum(self.W(X_W))

            X_D_dagger = xDot.clone().reshape([xDot.shape[0], xDot.shape[1], 1]).requires_grad_()
            D_dagger = torch.sum(self.D_dagger(X_D_dagger))
            self.fs = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0].reshape([x.shape[0], x.shape[1]]) \
                      + torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0].reshape([xDot.shape[0], xDot.shape[1]])
            del W, X_W, X_D_dagger, D_dagger 

    # Save the model
    def save(self, PATH):
        # mkdir
        Path(PATH).mkdir(parents=True, exist_ok=True)

        # Save all three neural networks
        torch.save(self, PATH + "/model.pth")
        torch.save(self.W.module.state_dict(), PATH + "/W.pth")
        torch.save(self.D.module.state_dict(), PATH + "/D.pth")
        torch.save(self.D_dagger.module.state_dict(), PATH + "/D_dagger.pth")

    # Load the nns
    def load_nns(self, PATH, mapDevice):
        self.W.module.load_state_dict(torch.load(PATH + "/W.pth", map_location=mapDevice))
        self.D.module.load_state_dict(torch.load(PATH + "/D.pth", map_location=mapDevice))
        self.D_dagger.module.load_state_dict(torch.load(PATH + "/D_dagger.pth", map_location=mapDevice))

        # Send to devices
        self.W.module.to(mapDevice)
        self.D.module.to(mapDevice)
        self.D_dagger.module.to(mapDevice)

# Class for Polynomial functions
class PN(nn.Module):
    def __init__(self, ProdOrder, logVFlag = False):
        super().__init__()
        self.ProdOrder = ProdOrder
        self.logVFlag = logVFlag

        if self.logVFlag:
            self.ProdOrder[0] += 1
        self.coefs = nn.Parameter(torch.rand(self.ProdOrder))
    
    def getPolyVal(self, x, coeffs):
        curVal = torch.zeros(x.shape, device=x.device)
        for curValIndex in range(0, len(coeffs)-1):
            curVal = (curVal + coeffs[curValIndex]) * x
        return (curVal + coeffs[-1])
    
    def forward(self, x):
        res = torch.ones(x.shape[0], device=x.device)
        if self.logVFlag:
            xin = torch.concat([x, torch.log(x[:, 0:1])], dim = 1)
            for idx in range(len(self.coefs)):
                res *= self.getPolyVal(xin[:, idx], self.coefs[idx, :])
        else:
            for idx in range(len(self.coefs)):
                res *= self.getPolyVal(x[:, idx], self.coefs[idx, :])
        return res


class PotentialsPolyCorrection:
    # Initialization of W and D
    def __init__(self, kwgsPolyPot):
        # self.dim_xi = kwgsPot["dim_xi"]
        self.dim_xi = 1 # Now only support 1 hidden variable

        self.logVFlag = False
        if "logVFlag" in kwgsPolyPot:
            self.logVFlag = kwgsPolyPot["logVFlag"]

        # Set up polynomial learning functions
        if "p_order_W" in kwgsPolyPot:
            self.p_order_W = kwgsPolyPot['p_order_W']
            self.W = PN([1, self.p_order_W])
            
            self.p_order_D = kwgsPolyPot['p_order_D']
            self.D = PN([self.dim_xi, self.p_order_D])

            self.p_order_D_dagger = kwgsPolyPot['p_order_D_dagger']
            self.D_dagger = PN([1 + self.dim_xi, self.p_order_D_dagger], self.logVFlag)
        else:
            self.p_order = kwgsPolyPot['p_order']
            self.W = PN([1, self.p_order])
            self.D = PN([self.dim_xi, self.p_order])
            self.D_dagger = PN([1 + self.dim_xi, self.p_order], self.logVFlag)

        ## DEBUG 
        print("PotentialsPolyCorrection self.logVFlag: ", self.logVFlag)
        print("PotentialsPolyCorrection self.D_dagger.coefs.shape[0]: ", self.D_dagger.coefs.shape[0])
        self.optim_W = optim.Adam(self.W.parameters(), lr=kwgsPolyPot["learning_rate"])
        self.optim_D = optim.Adam(self.D.parameters(), lr=kwgsPolyPot["learning_rate_D"])
        self.optim_D_dagger = optim.Adam(self.D_dagger.parameters(), lr=kwgsPolyPot["learning_rate_D_dagger"])
        
        # Device
        self.device = kwgsPolyPot["device"]
        
        # Multi-GPU data parallel
        self.W = nn.DataParallel(self.W)
        self.D = nn.DataParallel(self.D)
        self.D_dagger = nn.DataParallel(self.D_dagger)
        
        self.W.to(self.device)
        self.D.to(self.device)
        self.D_dagger.to(self.device)
        
    # Calculate f 
    def calf(self, x, xDot, t):
        # Initialize Vs
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        # xis[:, :, :] = 1. 
        
        
        # Loop through time steps
        
        if self.dim_xi > 0:
            xiNext = torch.zeros([batch_size, self.dim_xi], requires_grad=True, device=self.device)
            # xi0 = torch.zeros([batch_size, self.dim_xi], requires_grad=True, device=self.device)
            
            # List of fs
            list_fs = []
            # list_xis = [xi0]
            
            for idx in range(x.shape[1]):
                # Load xi_curr
                xiCurr = xiNext

                # f = \partial W / \partial V
                X_D_dagger = torch.concat([xDot[:, idx:idx + 1], xiCurr], dim = 1).requires_grad_()
                # X_W.to(self.device)
                D_dagger = torch.sum(self.D_dagger(X_D_dagger))

                this_piece = torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0]

                # Solve for \dot{\xi} + \partial W / \partial \xi = 0
                dD_daggerdXi = this_piece[:, 1:]
                dD_daggerdXDot = this_piece[:, 0:1]

                X_W = x[:, idx:idx+1].requires_grad_()
                W = torch.sum(self.W(X_W))
                dWdX = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0]

                list_fs.append(dD_daggerdXDot + dWdX.reshape([-1, 1]))

                # XiDot = dD^*/d\dot{d} (-dD^\dagger / dXi)
                if idx < x.shape[1] - 1:
                    this_input = -dD_daggerdXi.clone().requires_grad_()
                    D = torch.sum(self.D(this_input))
                    xiNext = xiCurr + torch.autograd.grad(outputs=D, inputs=this_input, create_graph=True)[0] * (t[:, idx + 1:idx + 2] - t[:, idx:idx + 1])
                    # list_xis.append(xiNext)
                    
                    del this_input, dD_daggerdXi, dD_daggerdXDot, W, X_W, dWdX, D, this_piece, X_D_dagger, D_dagger 
                else:
                    del dD_daggerdXi, dD_daggerdXDot, W, X_W, dWdX, this_piece, X_D_dagger, D_dagger
                    
            self.fs = torch.concat(list_fs, dim=1)
            del xiNext, xiCurr
            
        else:
            X_W = x.clone().reshape([x.shape[0], x.shape[1], 1]).requires_grad_()
            # print(X_W)
            W = torch.sum(self.W(X_W))

            X_D_dagger = xDot.clone().reshape([xDot.shape[0], xDot.shape[1], 1]).requires_grad_()
            D_dagger = torch.sum(self.D_dagger(X_D_dagger))
            self.fs = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0].reshape([x.shape[0], x.shape[1]]) \
                      + torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0].reshape([xDot.shape[0], xDot.shape[1]])
            del W, X_W, X_D_dagger, D_dagger 

    # Save the model
    def save(self, PATH):
        # mkdir
        Path(PATH).mkdir(parents=True, exist_ok=True)

        # Save all three neural networks
        torch.save(self, PATH + "/model.pth")
        torch.save(self.W.module.state_dict(), PATH + "/W.pth")
        torch.save(self.D.module.state_dict(), PATH + "/D.pth")
        torch.save(self.D_dagger.module.state_dict(), PATH + "/D_dagger.pth")

    # Load the nns
    def load_nns(self, PATH, mapDevice):
        self.W.module.load_state_dict(torch.load(PATH + "/W.pth", map_location=mapDevice))
        self.D.module.load_state_dict(torch.load(PATH + "/D.pth", map_location=mapDevice))
        self.D_dagger.module.load_state_dict(torch.load(PATH + "/D_dagger.pth", map_location=mapDevice))

        # Send to devices
        self.W.module.to(mapDevice)
        self.D.module.to(mapDevice)
        self.D_dagger.module.to(mapDevice)


class PotsCalXiXiDot:
    # Initialization of W and D
    def __init__(self, myWD):
        self.dim_xi = myWD.dim_xi
        self.W = myWD.W
        self.D = myWD.D
        self.D_dagger = myWD.D_dagger
        
        # Device
        # self.device = "cpu"
        self.W.to('cpu')
        self.D.to('cpu')
        self.D_dagger.to('cpu')
        self.fs = []
        self.xis = []
        self.xiDots = []
        self.Dins = []

    # Calculate f 
    def calf(self, x, xDot, t):
        # Initialize Vs
        batch_size = x.shape[0]
        x = x.to("cpu")
        xDot = xDot.to("cpu")
        t = t.to("cpu")

        # Loop through time steps
        if self.dim_xi > 0:
            xi0 = torch.zeros([batch_size, self.dim_xi], requires_grad=True, device='cpu')
            
            # List of fs
            list_fs = []
            list_xis = [xi0]
            list_xiDots = []
            list_Dins = []

            for idx in range(x.shape[1]):
                # f = \partial W / \partial V
                X_D_dagger = torch.concat([xDot[:, idx:idx + 1], list_xis[-1]], dim = 1).requires_grad_()
                # X_W.to(self.device)
                D_dagger = torch.sum(self.D_dagger(X_D_dagger))

                this_piece = torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0]

                # Solve for \dot{\xi} + \partial W / \partial \xi = 0
                dD_daggerdXi = this_piece[:, 1:]
                dD_daggerdXDot = this_piece[:, 0:1]

                X_W = x[:, idx:idx+1].requires_grad_()
                W = torch.sum(self.W(X_W))
                dWdX = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0]

                list_fs.append(dD_daggerdXDot + dWdX.reshape([-1, 1]))

                # XiDot = dD^*/d\dot{d} (-dD^\dagger / dXi)
                
                this_input = -dD_daggerdXi.clone().requires_grad_()
                D = torch.sum(self.D(this_input))
                xiDot = torch.autograd.grad(outputs=D, inputs=this_input, create_graph=True)[0]
                list_xiDots.append(xiDot)
                list_Dins.append(this_input)

                if idx < x.shape[1] - 1:
                    xiNext = list_xis[-1] + xiDot * (t[:, idx + 1:idx + 2] - t[:, idx:idx + 1])
                    list_xis.append(xiNext)
                    del xiNext 
                    
                del this_input, dD_daggerdXi, dD_daggerdXDot, W, X_W, dWdX, D, this_piece, X_D_dagger, D_dagger 
                    
                self.fs = torch.concat(list_fs, dim=1)
                if self.dim_xi == 1:
                    self.xis = torch.concat(list_xis, dim=1)
                    self.xiDots = torch.concat(list_xiDots, dim=1)
                    self.Dins = torch.concat(list_Dins, dim=1)
        else:
            X_W = x.clone().reshape([x.shape[0], x.shape[1], 1]).requires_grad_()
            # print(X_W)
            W = torch.sum(self.W(X_W))

            X_D_dagger = xDot.clone().reshape([xDot.shape[0], xDot.shape[1], 1]).requires_grad_()
            D_dagger = torch.sum(self.D_dagger(X_D_dagger))
            self.fs = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0].reshape([x.shape[0], x.shape[1]]) \
                      + torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0].reshape([xDot.shape[0], xDot.shape[1]])
            del W, X_W, X_D_dagger, D_dagger 


# f = f(x, \dot{x}, \xi), \dot{\xi} = g(x, \dot{x}, \xi)
class FricCorrection:
    # Initialization of W and D
    def __init__(self, kwgsPot):
        self.dim_xi = kwgsPot["dim_xi"]
        self.NNs_F = kwgsPot["NNs_W"]
        self.NNs_G = kwgsPot["NNs_D"]
        # self.NNs_D_dagger = kwgsPot["NNs_D_dagger"]
        self.F = PP(self.NNs_F, input_dim = 2 + self.dim_xi, output_dim = 1)
        self.G = PP(self.NNs_G, input_dim = 2 + self.dim_xi, output_dim = self.dim_xi)
        # self.D_dagger = PP(self.NNs_D_dagger, input_dim = 1 + self.dim_xi, output_dim = 1)
        self.optim_F = optim.Adam(self.F.parameters(), lr=kwgsPot["learning_rate"])
        self.optim_G = optim.Adam(self.G.parameters(), lr=kwgsPot["learning_rate_D"])
        # self.optim_D_dagger = optim.Adam(self.D_dagger.parameters(), lr=kwgsPot["learning_rate_D_dagger"])
        
        # Device
        self.device = kwgsPot["device"]
        self.F.to(self.device)
        self.G.to(self.device)
        # self.D_dagger.to(self.device)
        
    # Calculate f 
    def calf(self, x, xDot, t):
        # Initialize Vs
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        # xis[:, :, :] = 1. 
        
        
        # Loop through time steps
        
        if self.dim_xi > 0:
            xi0 = torch.zeros([batch_size, self.dim_xi], device=self.device)
            
            # List of fs
            list_fs = []
            list_xis = [xi0]
            
            for idx in range(x.shape[1]):
                # f = \partial W / \partial V
                X = torch.concat([x[:, idx:idx + 1], xDot[:, idx:idx + 1], list_xis[-1]], dim = 1)
                list_fs.append(self.F(X).reshape([-1, 1]))

                # xiDot = self.G(X)
                # list_xis.append(list_xis[-1] + xiDot * (t[:, idx + 1:idx + 2] - t[:, idx:idx + 1]))
                
                if idx < x.shape[1] - 1:
                    xiDot = self.G(X)
                    list_xis.append(list_xis[-1] + xiDot * (t[:, idx + 1:idx + 2] - t[:, idx:idx + 1]))
                    # print("list_xis[-1], xiDot shapes: ", list_xis[-1].shape, xiDot.shape)
                    
            del X, xiDot
            self.fs = torch.concat(list_fs, dim=1)

        else:
            X = torch.stack([x, xDot], dim=2)
            # print(X_W)
            self.fs = self.F(X).reshape([batch_size, time_steps])
            # print("self.fs.shape: ", self.fs.shape)
            del X


# Define loss functions given fs_targ, fs. 
def Loss(fs_targ, fs, ts, fOffSet, p = 2):
    err = torch.trapz(torch.abs(fs_targ - fs) ** p, ts, dim = 1) / torch.trapz(torch.abs(fs_targ - fOffSet) ** p, ts, dim = 1)
    err = torch.pow(err, 1. / p)
    return torch.sum(err)

# Training for one epoch
def train1Epoch(data_loader, loss_fn, myPot, p, fOffSet, update_weights=True):
    # Record of losses for each batch
    Losses = []
    device=myPot.device
    
    # Enumerate over data_loader
    for idx, (Xs, XDots, ts, fs_targ) in enumerate(data_loader):
        # Send shits to GPU
        Xs = Xs.to(device)
        XDots = XDots.to(device)
        ts = ts.to(device)
        fs_targ = fs_targ.to(device)
        
        # Refresh the optimizers
        if hasattr(myPot, 'optim_W'):
            myPot.optim_W.zero_grad()
        
        if hasattr(myPot, 'optim_D'):
            myPot.optim_D.zero_grad()
        
        if hasattr(myPot, 'optim_D_dagger'):
            myPot.optim_D_dagger.zero_grad()
        
        if hasattr(myPot, 'optim_F'):
            myPot.optim_F.zero_grad()

        if hasattr(myPot, 'optim_G'):
            myPot.optim_G.zero_grad()
         
        ## DEBUG LINE CHECK DEVICES
        # print("Xs.device: ", Xs.device)
        # print("Xs[:, 0:1].device: ", Xs[:, 0:1].device)
        
        # Compute loss
        myPot.calf(Xs, XDots, ts)
        loss = loss_fn(fs_targ, myPot.fs, ts, fOffSet, p)
        Losses.append(loss)
        
        # Update the model parameters
        if update_weights:
            loss.backward()
            
            if hasattr(myPot, 'optim_W'):
                myPot.optim_W.step()
        
            if hasattr(myPot, 'optim_D'):
                myPot.optim_D.step()
            
            if hasattr(myPot, 'optim_D_dagger'):
                myPot.optim_D_dagger.step()

            if hasattr(myPot, 'optim_F'):
                myPot.optim_F.step()

            if hasattr(myPot, 'optim_G'):
                myPot.optim_G.step()
         
        
    res = sum(Losses) / len(data_loader.dataset)
    res = res.to("cpu")

    # print("Memory before del in train1Epoch: ")
    # memory_stats()

    del Xs, XDots, ts, fs_targ, Losses
    torch.cuda.empty_cache()

    # print("Memory after del in train1Epoch: ")
    # memory_stats()
    return res

## Spring slider
# Function to plot a sequence compared between R & S and NN models
def plotGenVXFric(VV, tt, t, Vs, xs, Frics):
    # Plot Sequence of V(t) and N(t) given sample-index
    f, axs = plt.subplots(2, 2, figsize = (15, 15))

    ## data
    legends = ["R \& S", "NN"]
    lws = [4.0, 2.0]

    # Plot genVVtt
    axs[0][0].tick_params(axis='both', which='major', labelsize=20)
    axs[0][0].tick_params(axis='both', which='major', labelsize=20)
    axs[0][0].plot(tt, VV, linewidth=4.0, label="Pulling V")
    axs[0][0].set_xlabel('Time [s]', fontsize=20)
    axs[0][0].set_ylabel('Pulling $V(t)\  \mathrm{[m/s]}$', fontsize=20)
    axs[0][0].grid()

    # Plot v_1(t)
    axs[0][1].tick_params(axis='both', which='major', labelsize=20)
    for i in range(len(Vs)):
        axs[0][1].semilogy(t, Vs[i], linewidth=lws[i], label=legends[i])
    axs[0][1].set_xlabel('Time [s]', fontsize=20)
    axs[0][1].set_ylabel('Slip rate $V(t)\ \mathrm{[m/s]}$', fontsize=20)
    axs[0][1].grid()

    # Plot x(t)
    axs[1][0].tick_params(axis='both', which='major', labelsize=20)
    axs[1][0].tick_params(axis='both', which='major', labelsize=20)
    for i in range(len(xs)):
        axs[1][0].plot(t, xs[i], linewidth=lws[i], label=legends[i])
    axs[1][0].set_xlabel('Time [s]', fontsize=20)
    axs[1][0].set_ylabel('Slip $x(t)\  \mathrm{[m]}$', fontsize=20)
    axs[1][0].grid()

    # Plot friction coefficient(t)
    axs[1][1].tick_params(axis='both', which='major', labelsize=20)
    for i in range(len(Frics)):
        axs[1][1].plot(t, Frics[i], linewidth=lws[i], label=legends[i])
    axs[1][1].set_xlabel('Time [s]', fontsize=20)
    axs[1][1].set_ylabel('Fric Coeff.', fontsize=20)
    axs[1][1].legend(fontsize=20, loc='best')
    axs[1][1].grid()

    return f, axs

def load_model(modelPrefix, mapDevice=torch.device("cpu"), dim_xi=1, dict_flag=False, NN_Flag = True):
    if dict_flag:
        
        PATH = "./model/" + modelPrefix + "_dimXi_" + str(dim_xi) + "_dict"
        myModel = torch.load(PATH + "/model.pth", map_location = mapDevice)
        myModel.device = mapDevice

        del myModel.W, myModel.D, myModel.D_dagger
        
        if NN_Flag == True:
            myModel.W = PP(myModel.NNs_W, input_dim = 1, output_dim = 1)
            myModel.D = PP(myModel.NNs_D, input_dim = myModel.dim_xi, output_dim = 1)
            myModel.D_dagger = PP(myModel.NNs_D_dagger, input_dim = 1 + myModel.dim_xi, output_dim = 1)
        else:
            p_order_W = myModel.W.module.ProdOrder[1]
            p_order_D = myModel.D.module.ProdOrder[1]
            p_order_D_dagger = myModel.D_dagger.module.ProdOrder[1]
            myModel.W = PN([1, p_order_W])
            myModel.D = PN([myModel.dim_xi, p_order_D])
            if hasattr(myModel, "logVFlag") and myModel.logVFlag:
                myModel.D_dagger = PN([1 + myModel.dim_xi, p_order_D_dagger], myModel.logVFlag)
            else:
                myModel.D_dagger = PN([1 + myModel.dim_xi, p_order_D_dagger])
                

        # Multi-GPU data parallel
        myModel.W = nn.DataParallel(myModel.W)
        myModel.D = nn.DataParallel(myModel.D)
        myModel.D_dagger = nn.DataParallel(myModel.D_dagger)

        myModel.W.module.load_state_dict(torch.load(PATH + "/W.pth", map_location=mapDevice))
        myModel.D.module.load_state_dict(torch.load(PATH + "/D.pth", map_location=mapDevice))
        myModel.D_dagger.module.load_state_dict(torch.load(PATH + "/D_dagger.pth", map_location=mapDevice))

        # Send to devices
        myModel.W = myModel.W.module.to(mapDevice)
        myModel.D = myModel.D.module.to(mapDevice)
        myModel.D_dagger = myModel.D_dagger.module.to(mapDevice)
    else:
        myModel = torch.load("./model/" + modelPrefix + "_dimXi_" + str(dim_xi) + ".pth", map_location = mapDevice)
        myModel.device = mapDevice
    return myModel