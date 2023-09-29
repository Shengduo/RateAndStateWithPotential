import torch
import numpy as np
import torchdiffeq
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

"""
Class RateStateParams, manages data of rate-and-state friction with flash-heating effect, contains 
    Data:
        fr: Reference friction coefficient
        a: Rate-and-state parameter a
        b: Rate-and-state parameter b
        DRS: Characteristic slip distance
        Vw: Flash-heating slip rate
        fw: Flash-heating friction coefficient
        
"""
class RateStateParams:
    # Constructor
    def __init__(self, fr = 0.6, Vr = 1.e-6, a = 0.016, b = 0.011, DRS = 1.e-6, Vw = 1.0e6, fw = 0.2):
        self.fr = fr
        self.Vr = Vr
        self.a = a
        self.b = b
        self.DRS = DRS
        self.Vw = Vw
        self.fw = fw
        
    # Output the information of this class
    def print_info(self):
        print("-" * 20, " Rate-and-state parameters ", "-"*20)
        print('fr:       ', self.fr)
        print('a:        ', self.a)
        print('b:        ', self.b)
        print('Vr:       ', self.Vr)
        print('DRS:      ', self.DRS)
        print('Vw:       ', self.Vw)
        print('fw:       ', self.fw)


"""
Class TimeSequenceGen, container for a Generated time sequence containing 
    Data:
        t [number of time points]: Tensor for time stamps, prescribed
        N [number of time points]: Tensor for normal stress, prescribed
        Ds [number of time points]: Tensor for slip rate, prescribed
        s [number of time points]: Tensor for slip, prescribed
        theta [number of hidden variables, number of time points]: Tensor for hidden variables, computed
        params : Class for parameters
   
    Method:
        __init__ : Constructor
        DthetaDt : Evolution function for hidden variables theta
        calTau : Calculate the function of shear stress Tau
        
"""
class TimeSequenceGen:
    # Constructor
    def __init__(self, t, N, Ds, params, regularizedFrictionLaw):
        # Load the parameters
        self.t = t
        self.N = N
        self.Ds = Ds
        self.s = torch.zeros(Ds.shape, dtype = torch.float64)
        self.s[:, 1:] = torch.cumulative_trapezoid(Ds, x = t)
        self.params = params
        self.regFlag = regularizedFrictionLaw
        
        # Calculate theta history
        self.theta = torch.zeros(Ds.shape, dtype = torch.float64)
        
        # Compute the interpolation for slip rate Ds
        self.t_temp = torch.concat([self.t, torch.tensor([self.t[-1] + 1.0])], 0)
        self.Ds_temp = torch.concat([self.Ds, self.Ds[:, -1].reshape([-1, 1])], 1)
        self.DsAtT = interp1d(self.t_temp, self.Ds_temp, kind="cubic")
        
        # Evolve theta(t)
        self.theta = torch.transpose(self.calTheta(), 0, 1)

        # Finish computing tau
        self.tau = self.calTau()
    
    # Function DthetaDt, defines DthetaDt as a function of temporally local values
    def DthetaDt(self, t, theta):
        # print('t = ', t)
        DthetaDt = 1. - (torch.tensor(self.DsAtT(t)) * theta / self.params.DRS)
        return DthetaDt
    
    # Function calTau, calculates shear traction tau, suingregularized rate-and-state formulation
    def calTau(self):
        # Set aliases for rate-and-state parameters
        a = self.params.a
        b = self.params.b
        Vr = self.params.Vr
        DRS = self.params.DRS
        fr = self.params.fr
        if self.regFlag:
            tau = self.N * a * torch.asinh(
                   self.Ds / 2. / Vr * torch.exp((fr + b * torch.log(Vr * self.theta / DRS)) / a)
                   )
        else :
            tau = self.N * (fr + a * torch.log(self.Ds / Vr) + b * (self.theta * Vr / DRS))
            
        return tau
    
    # Calculate theta using s(t), Ds(t), theta(t) and params
    def calTheta(self, theta0 = 1.0):
        theta = odeint(self.DthetaDt, theta0 * torch.ones(self.Ds.shape[0], dtype = torch.float64), self.t, 
                       rtol = 1.e-10, atol = 1.e-12, method = 'dopri8')
        return theta
    
"""
Function generateVAndN, generate V and N sequences. 
    Inputs: #-----------------------------------------------------------------------------------------
        N_seq: Number of sequences to be generated
        N_grid_points: Number of grid points in [0, T]
        n_Fourier: Number of Fourier terms in the generating functions
        T: Maximum time
        target_logV_range: The range of logV
        target_N_range: The range of normal stress
        
    Outputs: #----------------------------------------------------------------------------------------
        NAll [N_seq, N_grid_points]: Generated sample sequences of N(t)
        VAll [N_seq, N_grid_points]: Generated sample sequences of V(t)
        
"""

def generateTVAndN(N_seq = 10_000, 
                  N_grid_points = 1_000, 
                  n_Fourier = 16, 
                  T = 100.e-6, 
                  target_logV_range = [-15., 2.], 
                  target_N_range = [1., 12.]):
    
    # Generate time grid (uniform)
    t = torch.linspace(0., T, N_grid_points, dtype = torch.float64)
    
    # Pre-calculate sin(k pi/T t) and cos(k pi/T t)
    K = torch.linspace(0, n_Fourier - 1, n_Fourier, dtype = torch.float64)
    kPitOverT = K.reshape([-1, 1]) * torch.pi / T * t


    # Generate N(t) sequences in MPa, N_seq times
    NfSin = (torch.rand([N_seq, n_Fourier], dtype = torch.float64) - 0.5) * 5.
    NfCos = (torch.rand([N_seq, n_Fourier], dtype = torch.float64) - 0.5) * 5.

    # All N_seq normal tractions
    NAll = torch.matmul(NfSin, torch.sin(kPitOverT)) + torch.matmul(NfCos, torch.cos(kPitOverT))

    # Generate logV(t) sequences in m/s, N_seq times
    VfSin = (torch.rand([N_seq, n_Fourier], dtype = torch.float64) - 0.5) * 10.
    VfCos = (torch.rand([N_seq, n_Fourier], dtype = torch.float64) - 0.5) * 10.

    # All N_seq normal tractions
    VAll = torch.matmul(VfSin, torch.sin(kPitOverT)) + torch.matmul(VfCos, torch.cos(kPitOverT)) - 7.


    ## Rescale VAll, NAll into target range
    # ---------------------------------------------------------------------------------------------------------
    minVAll = torch.min(VAll, axis=1).values
    maxVAll = torch.max(VAll, axis=1).values
    VAll = VAll / (maxVAll - minVAll).reshape([-1, 1]) * (target_logV_range[1] - target_logV_range[0]) 
    # VAll -= (torch.min(VAll, axis=1).values - target_logV_range[0]).reshape([-1, 1])
    # VAll = torch.clip(VAll, min = target_logV_range[0], max = target_logV_range[1])
    print('Min, max of logV: ', torch.min(VAll).item(), torch.max(VAll).item())
    VAll = torch.pow(10., VAll)

    # NAll = torch.clip(NAll, min = target_N_range[0], max = target_N_range[1])
    minNAll = torch.min(NAll, axis=1).values
    maxNAll = torch.max(NAll, axis=1).values
    NAll = NAll / (maxNAll - minNAll).reshape([-1, 1]) * (target_N_range[1] - target_N_range[0]) 
    NAll -= (torch.min(NAll, axis=1).values - target_N_range[0]).reshape([-1, 1])

    print('Min, max of N: ', torch.min(NAll).item(), torch.max(NAll).item())

    return t, NAll, VAll

"""
Function generateVAndNPiecewiseConst, generate V and N sequences. 
    Inputs: #-----------------------------------------------------------------------------------------
        N_seq: Number of sequences to be generated
        N_grid_points: Number of grid points in [0, T]
        n_Fourier: Number of Fourier terms in the generating functions
        T: Maximum time
        target_logV_range: The range of logV
        target_N_range: The range of normal stress
        
    Outputs: #----------------------------------------------------------------------------------------
        NAll [N_seq, N_grid_points]: Generated sample sequences of N(t)
        VAll [N_seq, N_grid_points]: Generated sample sequences of V(t)
        
"""

def generateTVAndNPiecewiseConst(N_seq = 10_000, 
                  N_grid_points = 1_000, 
                  n_intervals = 16, 
                  T = 100.e-6, 
                  target_logV_range = [-15., 2.], 
                  target_N_range = [1., 12.]):
    
    # Generate time grid (uniform)
    t = torch.linspace(0., T, N_grid_points, dtype = torch.float64)
    
    # Pre-calculate sin(k pi/T t) and cos(k pi/T t)
    K = torch.linspace(0, n_intervals - 1, n_intervals, dtype = torch.float64)
    kPitOverT = K.reshape([-1, 1]) * torch.pi / T * t


    # Generate N(t) sequences in MPa, N_seq times
    NfSin = (torch.rand([N_seq, n_intervals], dtype = torch.float64) - 0.5) * 5.
    NfCos = (torch.rand([N_seq, n_intervals], dtype = torch.float64) - 0.5) * 5.

    # All N_seq normal tractions
    NAll = torch.matmul(NfSin, torch.sin(kPitOverT)) + torch.matmul(NfCos, torch.cos(kPitOverT))

    # Generate logV(t) sequences in m/s, N_seq times
    VLevels = torch.rand([N_seq, n_intervals], dtype = torch.float64) \
              * (target_logV_range[1] - target_logV_range[0]) + target_logV_range[0]
               
    VLevels = torch.cat([torch.zeros([N_seq, 1], dtype = torch.float64), VLevels], 1)
    
    print('Min, max of VLevels: ', torch.min(VLevels).item(), torch.max(VLevels).item())
    
    VPoints = torch.rand([N_seq, n_intervals - 1], dtype = torch.float64) * T 
    VPoints = torch.sort(VPoints, 1).values
    VPoints = torch.cat([torch.zeros([N_seq, 1], dtype = torch.float64), VPoints], 1)
    
    coarseFactor = 10
    tt = torch.linspace(0. - 10.e-6, T + 10.e-6, N_grid_points // coarseFactor, dtype = torch.float64)
    
    VAlltt = torch.zeros([N_seq, N_grid_points // coarseFactor], dtype = torch.float64)
    for i in range(n_intervals):
        VAlltt = VAlltt + torch.heaviside(tt - (VPoints[:, i].reshape([-1, 1])), torch.tensor(0.0, dtype = torch.float64)) \
                * ((VLevels[:, i + 1] - VLevels[:, i]).reshape([-1, 1])) 
    
    VAllf = interp1d(tt, VAlltt)
    VAll = torch.tensor(VAllf(t), dtype = torch.float64)
    # All N_seq normal tractions
    # VAll = torch.matmul(VfSin, torch.sin(kPitOverT)) + torch.matmul(VfCos, torch.cos(kPitOverT)) - 7.


    ## Rescale VAll, NAll into target range
    # ---------------------------------------------------------------------------------------------------------
    # minVAll = torch.min(VAll, axis=1).values
    # maxVAll = torch.max(VAll, axis=1).values
    # VAll = VAll / (maxVAll - minVAll).reshape([-1, 1]) * (target_logV_range[1] - target_logV_range[0]) 
    # VAll -= (torch.min(VAll, axis=1).values - target_logV_range[0]).reshape([-1, 1])
    # VAll = torch.clip(VAll, min = target_logV_range[0], max = target_logV_range[1])
    print('Min, max of logV: ', torch.min(VAll).item(), torch.max(VAll).item())
    VAll = torch.pow(10., VAll)

    # NAll = torch.clip(NAll, min = target_N_range[0], max = target_N_range[1])
    minNAll = torch.min(NAll, axis=1).values
    maxNAll = torch.max(NAll, axis=1).values
    NAll = NAll / (maxNAll - minNAll).reshape([-1, 1]) * (target_N_range[1] - target_N_range[0]) 
    NAll -= (torch.min(NAll, axis=1).values - target_N_range[0]).reshape([-1, 1])

    print('Min, max of N: ', torch.min(NAll).item(), torch.max(NAll).item())

    return t, NAll, VAll

# Define the function as a wrapper for generate tau and theta
def getSTauAndTheta(t, N, V, params, regularizedFrictionLaw = True):
    mySeq = TimeSequenceGen(t, N, V, params, regularizedFrictionLaw)
    return mySeq.s, mySeq.tau, mySeq.theta


## Generate and save the data
import time
# Generate the sequences
filename = "./data/sequencePieceConst1019_UnReg.pt"
N_samples = 10000
N_gridPts = 1_000

t, NAll, VAll = generateTVAndNPiecewiseConst(N_seq = N_samples, N_grid_points = N_gridPts)
RSparams = RateStateParams()
RSparams.print_info()

# Compute tau and theta
st_time = time.time()
sAll, tauAll, thetaAll = getSTauAndTheta(t, NAll, VAll, RSparams, False)
timeConsumed = time.time() - st_time
print('Time consumed (single core): ', timeConsumed, ' s')

# Save to file
seqObj = {'SUnReg' : sAll, 'N' : NAll, 'V' : VAll, 'tauUnReg' : tauAll, 'thetaAllUnReg' : thetaAll, 't' : t}
torch.save(seqObj, filename)