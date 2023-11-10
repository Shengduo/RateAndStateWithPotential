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
import torch.optim as optim

# Memory management on GPU
import gc

# Import time
import time

# Testify whether GPU is available
print("Cuda is available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Device is: ", device)

def memory_stats():
    print("Memory allocated: ", torch.cuda.memory_allocated()/1024**2)
    print("Memory cached: ", torch.cuda.memory_reserved()/1024**2)
memory_stats()

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
            self.activation,
        )
        
        for i in range(len(NNs) - 1):
            self.fc.append(nn.Linear(NNs[i], NNs[i + 1]))
            self.fc.append(self.activation)
        
        self.fc.append(nn.Linear(NNs[-1], output_dim))
        self.fc.append(self.activation)
    
    # Forward function
    def forward(self, x):
        return self.fc(x)

## Data generation 
from DataGeneration import generateSamples, genVVtt
import os

generating_flag = False
# kwgs = {
#     "beta" : [0.006, 0.012, 1. / 1.e10, 0.58], 
#     "totalNofSeqs" : 100, # 1024 * 16, 
#     "NofIntervalsRange" : [4, 6], #[5, 11], 
#     "VVRange" : [-3, 1], 
#     "VVLenRange" : [3, 4], 
#     "theta0" : 0.1, 
#     "prefix" : "Trial1108_bigDRS", 
#     "NofVVSteps" : 10, 
# }

kwgs = {
    "beta" : [0.006, 0.012, 1. / 1.e12, 0.58], 
    "totalNofSeqs" : 100, # 1024 * 16, 
    "NofIntervalsRange" : [3, 5], #[5, 11], 
    "VVRange" : [-3, 1], 
    "VVLenRange" : [3, 4], 
    "Tmax" : 2.0, 
    "nTSteps" : 100, 
    "theta0" : 0.1, 
    "prefix" : "Trial1108_bigDRS_Burigede", 
    "NofVVSteps" : 10, 
}

# Generate / load data
dataFile = "./data/" + kwgs["prefix"] + ".pt"

if generating_flag or not(os.path.isfile(dataFile)):
    print("Generating data")
    generateSamples(kwgs)

shit = torch.load(dataFile)
Vs = shit["Vs"]
thetas = shit["thetas"]
fs = shit["fs"]
ts = shit["ts"]

# # Stack data as
# Vs = torch.stack(Vs)
# thetas = torch.stack(thetas)
# fs = torch.stack(fs)
# ts = torch.stack(ts)

# Now Vs and ts have fixed length
print("Vs.shape: ", Vs.shape)
print("thetas.shape: ", thetas.shape)
print("fs.shape: ", fs.shape)
print("ts.shape: ", ts.shape)

# Calculate Xs
Xs = torch.zeros(Vs.shape)
Xs[:, 1:] = torch.cumulative_trapezoid(Xs, ts)
print("Xs.shape: ", Xs.shape)

## Calculate f
# Different Potentials with D correction
class PotentialsFricCorrection:
    # Initialization of W and D
    def __init__(self, kwgsPot):
        self.dim_xi = kwgsPot["dim_xi"]
        self.NNs_W = kwgsPot["NNs_W"]
        self.NNs_D = kwgsPot["NNs_D"]
        self.NNs_D_dagger = kwgsPot["NNs_D_dagger"]
        self.W = PP(self.NNs_W, input_dim = 1 + self.dim_xi, output_dim = 1)
        self.D = PP(self.NNs_D, input_dim = self.dim_xi, output_dim = 1)
        self.D_dagger = PP(self.NNs_D_dagger, input_dim = 1, output_dim = 1)
        self.optim_W = optim.Adam(self.W.parameters(), lr=kwgsPot["learning_rate"])
        self.optim_D = optim.Adam(self.D.parameters(), lr=kwgsPot["learning_rate_D"])
        self.optim_D_dagger = optim.Adam(self.D_dagger.parameters(), lr=kwgsPot["learning_rate_D_dagger"])
        
        # Device
        self.device = kwgsPot["device"]
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
            xi0 = torch.zeros([batch_size, self.dim_xi], requires_grad=True, device=self.device)
            
            # List of fs
            list_fs = []
            list_xis = [xi0]
            
            for idx in range(x.shape[1]):
                # f = \partial W / \partial V
                X_W = torch.concat([x[:, idx:idx + 1], list_xis[-1]], dim = 1).requires_grad_()
                # X_W.to(self.device)
                W = torch.sum(self.W(X_W))

                this_piece = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0]

                # Solve for \dot{\xi} + \partial W / \partial \xi = 0
                dWdXi = this_piece[:, 1:]

                X_D_dagger = xDot[:, idx:idx+1].requires_grad_()
                D_dagger = torch.sum(self.D_dagger(X_D_dagger))
                dD_daggerdXDot = torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0]

                list_fs.append(this_piece[:, 0:1] + dD_daggerdXDot.reshape([-1, 1]))

                # XiDot = -dWdXi
                if idx < x.shape[1] - 1:
                    this_input = -dWdXi.clone().requires_grad_()
                    D = torch.sum(self.D(this_input))
                    xiNext = list_xis[-1] + torch.autograd.grad(outputs=D, inputs=this_input, create_graph=True)[0] * (t[:, idx + 1:idx + 2] - t[:, idx:idx + 1])
                    list_xis.append(xiNext)
                    
                    del this_input, W, dWdXi, X_W, D, this_piece, dD_daggerdXDot, X_D_dagger, D_dagger 
                    
                self.fs = torch.concat(list_fs, dim=1)
        else:
            X_W = x.clone().reshape([x.shape[0], x.shape[1], 1]).requires_grad_()
            # print(X_W)
            W = torch.sum(self.W(X_W))

            X_D_dagger = xDot.clone().reshape([xDot.shape[0], xDot.shape[1], 1]).requires_grad_()
            D_dagger = torch.sum(self.D_dagger(X_D_dagger))
            self.fs = torch.autograd.grad(outputs=W, inputs=X_W, create_graph=True)[0].reshape([x.shape[0], x.shape[1]]) \
                      + torch.autograd.grad(outputs=D_dagger, inputs=X_D_dagger, create_graph=True)[0].reshape([xDot.shape[0], xDot.shape[1]])
            del W, X_W, X_D_dagger, D_dagger 
            
## Define loss function, training function, dataloaders
# Define loss functions given fs_targ, fs. 
def Loss(fs_targ, fs, ts, p = 2):
    err = torch.trapz(torch.abs(fs_targ - fs) ** p, ts, dim = 1) / torch.trapz(torch.abs(fs_targ) ** p, ts, dim = 1)
    err = torch.pow(err, 1. / p)
    return torch.sum(err)

# Training for one epoch
def train1Epoch(data_loader, loss_fn, myPot, p, update_weights=True):
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
        myPot.optim_W.zero_grad()
        
        if hasattr(myPot, 'optim_D'):
            myPot.optim_D.zero_grad()
        
        if hasattr(myPot, 'optim_D_dagger'):
            myPot.optim_D_dagger.zero_grad()
        
        ## DEBUG LINE CHECK DEVICES
        # print("Xs.device: ", Xs.device)
        # print("Xs[:, 0:1].device: ", Xs[:, 0:1].device)
        
        # Compute loss
        myPot.calf(Xs, XDots, ts)
        loss = loss_fn(fs_targ, myPot.fs, ts, p)
        Losses.append(loss)
        
        # Update the model parameters
        if update_weights:
            loss.backward()
            myPot.optim_W.step()
        
            if hasattr(myPot, 'optim_D'):
                myPot.optim_D.step()
            
            if hasattr(myPot, 'optim_D_dagger'):
                myPot.optim_D_dagger.step()

        
        
    res = sum(Losses) / len(data_loader.dataset)
    # print("Memory before del in train1Epoch: ")
    # memory_stats()

    del Xs, XDots, ts, fs_targ, Losses
    torch.cuda.empty_cache()

    # print("Memory after del in train1Epoch: ")
    # memory_stats()
    return res

# Initialize dataloaders
AllData = TensorDataset(
    Xs, 
    Vs, 
    ts, 
    fs
)

dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

train_len = int(len(Vs) * 0.8)
test_len = len(Vs) - train_len
trainDataset, testDataset = torch.utils.data.random_split(AllData, [train_len, test_len])

## Define optuna function
class OptunaObj:
    # Initialize
    def __init__(self, kwgs):
        self.dim_xi = kwgs['dim_xi']
        self.test_p = kwgs['test_p']
        self.test_batch_size = kwgs['test_batch_size']
        self.device = kwgs['device']
        self.training_dataset = kwgs['training_dataset']
        self.test_dataset = kwgs['test_dataset']
        self.modelSavePrefix = kwgs['modelSavePrefix']
        
        
    # Define the objective
    def objective(self, trial):
        # Dump for un-saved interuptions
        joblib.dump(this_study, "./data/1108_bigDRS_Burigede_WDsep_study_dim_xi_logV_DLeg_D_dagger_ELU1_" + str(self.dim_xi) + ".pkl")

        # Fixed parameters
        dim_xi = self.dim_xi
        NNs_D = []
        test_p = self.test_p
        test_batch_size = self.test_batch_size

        # Define NN for W
        W_layers = trial.suggest_int('W_layers', 2, 8)
        NNs_W = []
        for i in range(W_layers):
            this_W = 2 ** trial.suggest_int('W_layer_units_exponent_{}'.format(i), 4, 10)
            NNs_W.append(this_W)
            
        # Define NN for D
        D_layers = trial.suggest_int('D_layers', 2, 8)
        NNs_D = []
        for i in range(D_layers):
            this_D = 2 ** trial.suggest_int('D_layer_units_exponent_{}'.format(i), 4, 10)
            NNs_D.append(this_D)
        
        # Define NN for D_dagger
        D_dagger_layers = trial.suggest_int('D_dagger_layers', 2, 8)
        NNs_D_dagger = []
        for i in range(D_dagger_layers):
            this_D_dagger = 2 ** trial.suggest_int('D_dagger_layer_units_exponent_{}'.format(i), 4, 10)
            NNs_D_dagger.append(this_D_dagger)

        # Suggest learning rate
        learning_rate = 10 ** trial.suggest_float('log_learning_rate', -5., -1.)
        
        # Suggest learning rate for D
        learning_rate_D = 10 ** trial.suggest_float('log_learning_rate_D', -5., -1.)

        # Suggest learning rate for D
        learning_rate_D_dagger = 10 ** trial.suggest_float('log_learning_rate_D_dagger', -5., -1.)

        # Suggest batchsize
        training_batch_size = 2 ** trial.suggest_int('training_batch_size', 6, 12)

        # Suggest training p
        training_p = trial.suggest_int('training_p', 2, 8)

        # Suggest training epochs
        # training_epochs = 2 ** trial.suggest_int('training_epoch_exponents', 5, 9)
        training_epochs = 100

        params = {
            'dim_xi' : dim_xi, 
            'NNs_W' : NNs_W, 
            'NNs_D' : NNs_D, 
            'NNs_D_dagger' : NNs_D_dagger, 
            'learning_rate' : learning_rate, 
            'learning_rate_D' : learning_rate_D, 
            'learning_rate_D_dagger' : learning_rate_D_dagger,  
            'training_batch_size' : training_batch_size, 
            'training_p' : training_p, 
            'training_epochs' : training_epochs, 
            'device' : self.device,
        }
        
        
        # Set training dataloader
        training_batch_size = params['training_batch_size'] #1024
        trainDataLoader = DataLoader(
            self.training_dataset,
            batch_size = training_batch_size,
            shuffle = True,
        #    num_workers = 16,
            collate_fn = None,
            **dataloader_kwargs, 
        )

        # Set testing data loader
        testing_batch_size = self.test_batch_size # 256
        testDataLoader = DataLoader(
            self.test_dataset,
            batch_size = testing_batch_size,
            shuffle = True,
        #    num_workers = 16,
            collate_fn = None,
            **dataloader_kwargs, 
        )
        
        # Print out info
        print("-"*20, " Trial ", str(trial.number), " ", "-"*20, flush=True)
        st = time.time()
        print("Start timing: ")
        
        print("Parameters: ", flush=True)
        print(trial.params, flush=True)
        
        # Training
        myWD = PotentialsFricCorrection(params)
        for i in range(params['training_epochs']):
            avg_training_loss = train1Epoch(trainDataLoader, Loss, myWD, params['training_p'])
            
            if torch.isnan(avg_training_loss):
                break
            
            if i % 10 == 0:
                # avg_test_loss = train1Epoch(testDataLoader, Loss, myWD, self.test_p, update_weights=False)
                print("\t", "epoch ", str(i), "training error: ", str(avg_training_loss), flush=True)
                ## Print memory status
                print("Memory status after this epoch: ")
                memory_stats()
        
        # Return objective value for optuna
        res = train1Epoch(testDataLoader, Loss, myWD, self.test_p, update_weights=False)
        if len([this_study.trials]) == 1 or res < this_study.best_value:
            torch.save(myWD, './model/' + self.modelSavePrefix + '_model.pth')
        print("Time for this trial: ", time.time() - st)
        # Release GPU memory
        del myWD
        gc.collect()
        torch.cuda.empty_cache()
        
        ## Print memory status
        print("Memory status after this trial: ")
        memory_stats()
        
        return res
            
# Do a parametric study over number of hidden parameters
dim_xis = [0]
studys = []

# Tune parameters for dim_xi = 4
OptKwgs = {
    'dim_xi' : 4, 
    'test_p' : 2, 
    'test_batch_size' : len(testDataset), 
    'device' : device, 
    'training_dataset' : trainDataset, 
    'test_dataset' : testDataset, 
}


# Loop through all dim_xis
for dim_xi in dim_xis:
    OptKwgs['dim_xi'] = dim_xi
    OptKwgs['modelSavePrefix'] = kwgs['prefix'] + "_dim_xi_" + str(dim_xi)
    myOpt = OptunaObj(OptKwgs)
    this_study = optuna.create_study(direction='minimize')
    this_study.optimize(myOpt.objective, n_trials=50)
    studys.append(this_study)
