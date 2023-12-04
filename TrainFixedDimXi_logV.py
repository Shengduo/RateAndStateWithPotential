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

## Generating using Burigede's scheme
from DataGeneration import generateSamples_Burigede
import os
generating_flag = False
kwgs = {
    "beta" : [0.006, 0.012, 1. / 1.e12, 0.58], 
    "totalNofSeqs" : 100, # 1024 * 16, 
    "NofIntervalsRange" : [3, 5], #[5, 11], 
    "VVRange" : [-3, 1], 
    "VVLenRange" : [3, 4], 
    "Tmax" : 2.0, 
    "nTSteps" : 100, 
    "theta0" : 0.1, 
    # "prefix" : "Trial1108_bigDRS_Burigede", 
    "prefix" : "Trial1116_smallDRS_largeA", 
    "NofVVSteps" : 10, 
}

# Generate / load data
dataFile = "./data/" + kwgs["prefix"] + ".pt"
print("Data file: ", dataFile)

if generating_flag or not(os.path.isfile(dataFile)):
    print("Generating data")
    generateSamples_Burigede(kwgs)

shit = torch.load(dataFile)
Vs = shit["Vs"]
thetas = shit["thetas"]
fs = shit["fs"]
ts = shit["ts"]

# Now Vs and ts have fixed length
print("Vs.shape: ", Vs.shape)
print("thetas.shape: ", thetas.shape)
print("fs.shape: ", fs.shape)
print("ts.shape: ", ts.shape)
# Calculate Xs
Xs = torch.zeros(Vs.shape)
Xs[:, 1:] = torch.cumulative_trapezoid(Xs, ts)
print("Xs.shape: ", Xs.shape)

# Different Potentials with D correction
import torch.optim as optim
## Calculate f
# Different Potentials with D correction
from FrictionNNModels import FricCorrection, Loss, train1Epoch, PP, ReLUSquare

## Define loss function, training function, dataloaders
# Initialize dataloaders
AllData = TensorDataset(
    Xs, 
    Vs, 
    ts, 
    fs
)

dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_len = int(len(Vs) * 0.8)
test_len = len(Vs) - train_len

# Created using indices from 0 to train_size.
trainDataset = torch.utils.data.Subset(AllData, range(train_len))

# Created using indices from train_size to train_size + test_size.
testDataset = torch.utils.data.Subset(AllData, range(train_len, train_len + test_len))

def train(params, dim_xi):
    # Fixed parameters
    params['dim_xi'] = dim_xi
    training_dataset = params['training_dataset']
    test_dataset = params['test_dataset']

    # Set training dataloader
    training_batch_size = params['training_batch_size'] #1024
    trainDataLoader = DataLoader(
        training_dataset,
        batch_size = training_batch_size,
        shuffle = True,
    #    num_workers = 16,
        collate_fn = None,
        **dataloader_kwargs, 
    )

    # Set testing data loader
    testing_batch_size = len(test_dataset) # 256
    testDataLoader = DataLoader(
        test_dataset,
        batch_size = testing_batch_size,
        shuffle = True,
    #    num_workers = 16,
        collate_fn = None,
        **dataloader_kwargs, 
    )
    
    # Print out info
    st = time.time()
    print("Start timing: ")

    # Training
    myWD = FricCorrection(params)
    for i in range(params['training_epochs']):
        avg_training_loss = train1Epoch(trainDataLoader, Loss, myWD, params['training_p'])
        
        if torch.isnan(avg_training_loss):
            print("Training failed due to nan")
            return -1
        
        if i % 10 == 0:
            print("\t", "epoch ", str(i), "training error: ", str(avg_training_loss), flush=True)
            ## Print memory status
            print("Memory status after this epoch: ")
            memory_stats()
    
    # Return objective value for optuna
    res = train1Epoch(testDataLoader, Loss, myWD, params['test_p'], update_weights=False)
    print("res: ", res)
    print("Save this model!")
    torch.save(myWD, './model/' + params['modelSavePrefix'] + '_dimXi_{0}_FG_fixed.pth'.format(params['dim_xi']))
    print("Time for this training process: ", time.time() - st)
    
    # Release GPU memory
    del myWD
    gc.collect()
    torch.cuda.empty_cache()
    
    ## Print memory status
    print("Memory status after this trial: ")
    memory_stats()
    
    return res

# NN structures for training
params = {
        'dim_xi' : 0, 
        'NNs_W' : [256, 256, 256, 256, 256, 256, 256, 256], 
        'NNs_D' : [256, 256, 256, 256, 256, 256, 256, 256], 
        # 'NNs_D_dagger' : NNs_D_dagger, 
        'learning_rate' : 1.e-3, 
        'learning_rate_D' : 1.e-3, 
        # 'learning_rate_D_dagger' : learning_rate_D_dagger,  
        'training_batch_size' : 16, 
        'training_p' : 6, 
        'training_epochs' : 200, 
        'test_p' : 2, 
        'device' : device,
        'training_dataset' : trainDataset, 
        'test_dataset' : testDataset, 
        'modelSavePrefix' : kwgs['prefix'], 
    }

# Do a parametric study over number of hidden parameters
dim_xis = [0, 1, 2, 4, 8]
res_test_loss = []

# Loop through all dim_xis
for dim_xi in dim_xis:
    this_res = -1.
    while this_res == -1.:
        this_res = train(params, dim_xi)
    res_test_loss.append(this_res)

print("~-"*40, " Results Summary ", "-~"*40)
print("dim_xis: ", dim_xis)
print("res_test_loss :", res_test_loss)