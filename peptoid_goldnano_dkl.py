import shutil, os, time
from datetime import datetime
import numpy as np
import pandas as pd

import torch
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from models import SingleTaskGP, MultiTaskGP, SingleTaskDKL, MultiTaskDKL
from test_functions import ExperimentalTestFunction
from utils import *
from activephasemap.activelearn.simulators import PrabolicPhases, PhaseMappingExperiment
from activephasemap.np.neural_process import NeuralProcess
from activephasemap.activelearn.surrogates import update_npmodel
from activephasemap.activelearn.pipeline import ActiveLearningDataset
from activephasemap.activelearn.visuals import plot_npmodel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
# torch.manual_seed(20245)

ITERATION = 2
# hyper-parameters
MODEL_NAME = "dkl"
SIMULATOR = "parabolic"
BATCH_SIZE = 22

# Set up data and plot directories and manually create them to avoid overrides 
PLOT_DIR = './results/peptide_GNP/plots_dkl/'
SAVE_DIR = './results/peptide_GNP/'
EXPT_DIR = './experiments/dkl_expt/'

""" Set up pretrain NP model """
# Specify the Neural Process model
N_LATENT = 2
PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/results_pretrain/trained_model.pt"

np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

""" Set up design space bounds """
input_dim = 2 # dimension of design space
output_dim = N_LATENT
_bounds = [(0.0, 87.0), (0.0,11.0)] # specify actual bounds of the design variables
bounds = torch.tensor(_bounds).transpose(-1, -2).to(device)

""" Create a GP model class for surrogate """
if MODEL_NAME=="gp":
    model_args = {"model":"gp"}
elif MODEL_NAME=="dkl":
    model_args = {"model": "dkl",
    "regnet_dims": [64,64,64],
    "regnet_activation": "tanh",
    "pretrain_steps": 1000,
    "train_steps": 1000
    }

""" Helper functions """

def fit_npmodel(np_model, test_function, comps, spectra):
    data = ActiveLearningDataset(comps,spectra) 
    np_model_updated, _ = update_npmodel(test_function.sim.t, np_model, data) 

    return np_model_updated

def featurize_spectra(spectra_all):
    """ Obtain latent space embedding from spectra.
    """
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0.0, 1.0, n_domain)
    t = t.repeat(num_samples, 1).to(device)
    with torch.no_grad():
        z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2), spectra.unsqueeze(2)) 

    return z 

def run_iteration(comps_all, spectra_all):
    """ Perform a single iteration of active phasemapping.

    helper function to run a single iteration given 
    all the compositions and spectra obtained so far. 

    This function only takes in compositions and spectra collected
    so far as input and makes use of other variables defined in this file.
    This makes sure that we can run this function even on a fresh Hyak session.
    """
    _bounds = [(0.0, 1.0) for _ in range(input_dim)]
    standard_bounds = torch.tensor(_bounds).transpose(-1, -2).to(device)
    gp_model = initialize_model(MODEL_NAME, model_args, input_dim, output_dim, device) 

    train_x = torch.from_numpy(comps_all).to(device) 
    train_y = featurize_spectra(spectra_all)
    model_start = time.time()
    normalized_x = normalize(train_x, bounds).to(train_x)
    gp_model.fit_and_save(normalized_x, train_y, PLOT_DIR)
    model_end = time.time()
    print("fit time", model_end - model_start)

    acq_start = time.time()
    acquisition = construct_acqf_by_model(gp_model, normalized_x, train_y, output_dim)
    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, 
        standard_bounds, 
        q=BATCH_SIZE, 
        num_restarts=5, 
        raw_samples=16, 
        return_best_only=False,
        sequential=False,
        options={"batch_limit": 1, "maxiter": 10, "with_grad":True}
        )

    # calculate acquisition values after rounding
    acqf_values = acquisition(normalized_candidates)
    acq_end = time.time()
    print("acquisition time", acq_end - acq_start)

    best_index = acqf_values.max(dim=0).indices.item()
    candidates = unnormalize(normalized_candidates.detach(), bounds=bounds)
    new_x = candidates[best_index].to(train_x) 

    return new_x, gp_model, acquisition, train_x 

# Run phase mapping iterations

comps_new = np.load(EXPT_DIR+'comps_%d.npy'%ITERATION)
sim = PhaseMappingExperiment(ITERATION, EXPT_DIR, _bounds)
sim.generate()
sim.plot(PLOT_DIR+'train_spectra_%d.png'%ITERATION)

test_function = ExperimentalTestFunction(sim=sim, bounds=_bounds)

# assemble data for surrogate model training  
comps_all = test_function.sim.comps 
spectra_all = test_function.sim.spectra
print('Data shapes : ', comps_all.shape, spectra_all.shape)

# update the pretrained model from collected data
np_model = fit_npmodel(np_model, test_function, comps_all, spectra_all)
plot_npmodel(test_function.sim.t, N_LATENT, np_model, PLOT_DIR+'npmodel_itr_%d.png'%ITERATION)
# obtain new set of compositions to synthesize
new_x, gp_model, acquisition, train_x = run_iteration(comps_all, spectra_all)
np.save(SAVE_DIR+'dkl_new_%d.npy'%(ITERATION+1), new_x.cpu().numpy()) 

# visualize models trained so far
plot_iteration(ITERATION, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
plt.savefig(PLOT_DIR+'itr_%d.png'%ITERATION)
plt.close()  
plot_gpmodel_expt(test_function, gp_model, np_model, PLOT_DIR+'gpmodel_itr_%d.png'%ITERATION) 
plot_phasemap_pred(test_function, gp_model, np_model, PLOT_DIR+'compare_spectra_pred_%d.png'%ITERATION)

torch.save(train_x.cpu(), SAVE_DIR+'dkl_train_x.pt')
torch.save(gp_model.state_dict(), SAVE_DIR+'dkl_model.pt')