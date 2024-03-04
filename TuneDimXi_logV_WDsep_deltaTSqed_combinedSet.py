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

# Testify whether GPU is available
print("Cuda is available: ", torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    "prefix" : "Trial0216_combined_800", 
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
from FrictionNNModels import PotentialsFricCorrection, Loss, train1Epoch, PP, ReLUSquare

## Define loss function, training function, dataloaders
# Initialize dataloaders
AllData = {
    "Xs" : Xs, 
    "Vs" : Vs, 
    "ts" : ts, 
    "fs" : fs, 
}

dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

class OptunaObj:
    # Initialize
    def __init__(self, kwgs):
        self.dim_xi = kwgs['dim_xi']
        self.test_p = kwgs['test_p']
        self.device = kwgs['device']
        self.modelSavePrefix = kwgs['modelSavePrefix']
        self.bestValue = 10000000.

        if 'bestValue' in kwgs.keys():
            self.bestValue = kwgs['bestValue']
        
        self.fOffSet = kwgs['fOffSet']
        self.scaling_factor = kwgs['scaling_factor']

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
    def objective(self, trial):
        # Dump for un-saved interuptions
        # joblib.dump(this_study, "./jobs/" + self.modelSavePrefix + str(self.dim_xi) + ".pkl")

        # Fixed parameters
        dim_xi = self.dim_xi
        NNs_D = []

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
        print("-"*20, " Trial ", str(trial.number), " ", "-"*20, flush=True)
        st = time.time()
        print("Start timing: ")
        
        print("Parameters: ", flush=True)
        print(trial.params, flush=True)
        
        # Training
        myWD = PotentialsFricCorrection(params)
        for i in range(params['training_epochs']):
            avg_training_loss = train1Epoch(trainDataLoader, Loss, myWD, params['training_p'], 0.)
            
            if torch.isnan(avg_training_loss):
                break
            
            if i % 10 == 0:
                # avg_test_loss = train1Epoch(testDataLoader, Loss, myWD, self.test_p, update_weights=False)
                print("\t", "epoch ", str(i), "training error: ", str(avg_training_loss), flush=True)
                ## Print memory status
                print("Memory status after this epoch: ")
                memory_stats()
        
        # Return objective value for optuna
        res = train1Epoch(testDataLoader, Loss, myWD, self.test_p, 0., update_weights=False)
        
        if res < self.bestValue:
            print("res: ", res)
            print("self.bestValue: ", self.bestValue)
            print("Save this model!")

            saveDir = './model/' + self.modelSavePrefix + '_dimXi_{0}_dict'.format(self.dim_xi)
            myWD.save(saveDir)

        # Path(saveDir).makedir(parents=True, exist_ok=True)
        
        # torch.save(myWD.module.state_dict(), './model/' + self.modelSavePrefix + '_dimXi_{0}_dict.pth'.format(self.dim_xi))
        # if res < self.bestValue:
            self.bestValue = res

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
dim_xis = [1]
studys = []

# Tune parameters for dim_xi = 4
OptKwgs = {
    'dim_xi' : 4, 
    'test_p' : 2, 
    # 'test_batch_size' : len(testDataset), 
    'device' : device, 
    # 'training_dataset' : trainDataset, 
    # 'test_dataset' : testDataset, 
    'modelSavePrefix' : kwgs['prefix'],
    'fOffSet' : 0.5109, 
    'scaling_factor' : 50., 
    'AllData' : AllData, 
}

# Loop through all dim_xis
for dim_xi in dim_xis:
    # sqlite:///example.db
    this_study = optuna.create_study(direction='minimize', 
                                     storage="sqlite:///./jobs/{0}_{1}".format(kwgs['prefix'], dim_xi) + ".db", 
                                     study_name="my_study1", 
                                     load_if_exists=True)

    OptKwgs['dim_xi'] = dim_xi
    # print("this_study.best_trial.value: ", this_study.best_trial.value)

    try:
        OptKwgs['bestValue'] = this_study.best_trial.value
        print("Pruned database has best value {0}.".format(this_study.best_trial.value))
    except:
        OptKwgs['bestValue'] = 10000000.
        print("No pruned database has been founded.")


    myOpt = OptunaObj(OptKwgs)

    this_study.optimize(myOpt.objective, n_trials=129)
    studys.append(this_study)
