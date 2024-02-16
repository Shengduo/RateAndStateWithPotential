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
    # "prefix" : "Trial1116_smallDRS_largeA", 
    # "prefix" : "Trial1215_smallDRS_smallA", 
    # "prefix" : "Trial1215_smallDRS_Burigede", 
    "prefixs" : ["Trial0112_smallDRS_Burigede"], 
    "saveName" : "Trial0112_smallA_Burigede", 
    "pre_model" : "Trial0112_smallDRS_smallA", 
    "NofVVSteps" : 10, 
}

## Import packages and classes
# Different Potentials with D correction
import torch.optim as optim
## Calculate f
# Different Potentials with D correction
from FrictionNNModels import PotentialsFricCorrection, Loss, train1Epoch, PP, ReLUSquare

class TrainObj:
    # Initialize
    def __init__(self, kwgs):
        self.dim_xi = kwgs['dim_xi']
        self.test_p = kwgs['test_p']
        self.device = kwgs['device']
        self.fOffSet = kwgs['fOffSet']
        self.scaling_factor = kwgs['scaling_factor']

        # Load pre-model
        self.best_params = kwgs['best_params']
        self.pre_model = kwgs['pre_model']
        
        # Transform the input data
        AllData = TensorDataset(
            kwgs["AllData"]["Xs"], 
            kwgs["AllData"]["Vs"], 
            kwgs["AllData"]["ts"], 
            (kwgs["AllData"]["fs"] - self.fOffSet) * self.scaling_factor + self.fOffSet, 
        )

        self.dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

        train_len = int(len(Vs) * 0.8)
        test_len = len(Vs) - train_len

        # Created using indices from 0 to train_size.
        self.training_dataset = torch.utils.data.Subset(AllData, range(train_len))

        # Created using indices from train_size to train_size + test_size.
        self.test_dataset = torch.utils.data.Subset(AllData, range(train_len, train_len + test_len))


    # Define the objective
    def objective(self):
        # Suggest batchsize
        training_batch_size = 2 ** self.best_params['training_batch_size']

        # Suggest training p
        training_p = self.best_params['training_p']

        # Suggest training epochs
        # training_epochs = 2 ** trial.suggest_int('training_epoch_exponents', 5, 9)
        training_epochs = 100

        params = {
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
        testing_batch_size = len(self.test_dataset) # 256
        testDataLoader = DataLoader(
            self.test_dataset,
            batch_size = testing_batch_size,
            shuffle = True,
        #    num_workers = 16,
            collate_fn = None,
            **dataloader_kwargs, 
        )
        
        # Print out info
        
        # Training
        for i in range(params['training_epochs']):
            avg_training_loss = train1Epoch(trainDataLoader, Loss, self.pre_model, params['training_p'], 0.)
            
            if torch.isnan(avg_training_loss):
                break
            
            if i % 10 == 0:
                # avg_test_loss = train1Epoch(testDataLoader, Loss, myWD, self.test_p, update_weights=False)
                print("\t", "epoch ", str(i), "training error: ", str(avg_training_loss), flush=True)
                ## Print memory status
                print("Memory status after this epoch: ")
                memory_stats()
        
        # Return objective value for optuna
        res = train1Epoch(testDataLoader, Loss, self.pre_model, self.test_p, 0., update_weights=False)
        
        # Release GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        
        ## Print memory status
        print("Memory status after this trial: ")
        memory_stats()
        return res

# Load the pre_model
dim_xi = 1
pre_model = torch.load("./model/" + kwgs["pre_model"] + "_dimXi_" + str(dim_xi) + ".pth")

# Load best parameters for the pre_model
best_params = optuna.load_study(study_name="my_study", 
                                storage="sqlite:///./jobs/{0}_{1}.db".format(kwgs["pre_model"], dim_xi)).best_params

# Set train parameters
TrainKwgs = {
    'dim_xi' : dim_xi, 
    'test_p' : 2, 
    'device' : device, 
    'fOffSet' : 0.5109, 
    'scaling_factor' : 50., 
    'AllData' : dict(), 
    'best_params' : best_params, 
    'pre_model' : pre_model
}

# Train on datasets specified by prefixs
for idx, prefix in enumerate(kwgs["prefixs"]):
    # Load data
    dataFile = "./data/" + prefix + ".pt"
    print("======================== Online dataset {0} ========================".format(idx))
    print("Data file: ", dataFile)
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

    ## Define loss function, training function, dataloaders
    # Initialize dataloaders
    AllData = {
        "Xs" : Xs, 
        "Vs" : Vs, 
        "ts" : ts, 
        "fs" : fs, 
    }

    # Update trainKwgs with this dataset
    TrainKwgs['AllData'] = AllData
    dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    myOpt = TrainObj(TrainKwgs)

    # Triain myOpt
    myOpt.objective()

# Save the newly trained model
torch.save(pre_model, './model/' + kwgs["saveName"] + '_dimXi_{0}.pth'.format(dim_xi))