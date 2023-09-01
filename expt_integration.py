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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
torch.manual_seed(21548)

# hyper-parameters
MODEL_NAME = "gp"
SIMULATOR = "parabolic"
BATCH_SIZE = 4
N_INIT_POINTS = 5
N_ITERATIONS = 3

SAVE_DIR = './results/expt_data/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
PLOT_DIR = SAVE_DIR+'plots/'
os.makedirs(PLOT_DIR)

# setup a simulator to use as a proxy for experiment
if SIMULATOR=="parabolic":
    sim = PrabolicPhases(n_grid=100, use_random_warping=False, noise=True)
elif SIMULATOR=="gaussians":
    sim = GaussianPhases(n_grid=100, use_random_warping=False, noise=True)
elif SIMULATOR=="goldnano":
    dirloc = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/gold_nano_grid/"
    sim = GNPPhases(dirloc)

""" Set up pretrain NP model """
# Specify the Neural Process model
if SIMULATOR=="goldnano":
    N_LATENT = 2
    PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/results_pretrain/trained_model.pt"
else:
    N_LATENT = 3
    PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/phasemaps/pretrain/trained_model.pt"

np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

""" Set up design space bounds """
input_dim = 2 # dimension of design space
output_dim = N_LATENT
_bounds = [(1e-4, 1.0) for _ in range(input_dim)]
bounds = torch.tensor(_bounds).transpose(-1, -2).to(device)

""" Create a GP model class for surrogate """
if MODEL_NAME=="gp":
    model_args = {"model":"gp"}
elif MODEL_NAME=="dkl":
    model_args = {"model": "dkl",
    "regnet_dims": [32,32,32],
    "regnet_activation": "tanh",
    "pretrain_steps": 1000,
    "train_steps": 1000
    }

""" Helper functions """

def featurize_spectra(spectra_all):
    """ Obtain latent space embedding from spectra.
    """
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0, 1, n_domain)
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
    _bounds = [(float(1e-5), 1.0) for _ in range(input_dim)]
    standard_bounds = torch.tensor(_bounds).transpose(-1, -2).to(device)
    gp_model = initialize_model(MODEL_NAME, model_args, input_dim, output_dim, device) 

    train_x = torch.from_numpy(comps_all).to(device) 
    train_y = featurize_spectra(spectra_all)
    model_start = time.time()
    normalized_x = normalize(train_x, bounds).to(train_x)
    gp_model.fit_and_save(normalized_x, train_y, SAVE_DIR)
    model_end = time.time()
    print("fit time", model_end - model_start)

    acq_start = time.time()
    acquisition = construct_acqf_by_model(gp_model, normalized_x, train_y, output_dim)
    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, 
        standard_bounds, 
        q=BATCH_SIZE, 
        num_restarts=128, 
        raw_samples=512, 
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

# Set up a synthetic data emulating an experiment
init_x = initialize_points(bounds, N_INIT_POINTS, output_dim, device)
np.save(SAVE_DIR+'comps_0.npy', init_x.cpu().numpy())

for i in range(N_ITERATIONS):
    # the following line replicates the actual experiment
    # to test this code, we use a simulator that simply generates spectra
    # and saves them to folder named spectra_i.xlsx
    comps_new = np.load(SAVE_DIR+'comps_%d.npy'%i)
    # spectra should be saved in the following format
    spectra_new = np.zeros((len(comps_new), sim.n_domain))
    for j, cj in enumerate(comps_new):
        spectra_new[j,:] = sim.simulate(cj)

    df = pd.DataFrame(spectra_new)
    df.to_excel(SAVE_DIR+'spectra_%d.xlsx'%i, index=False)
    expt_sim = PhaseMappingExperiment(i, SAVE_DIR, _bounds)
    expt_sim.generate()
    test_function = ExperimentalTestFunction(sim=expt_sim, bounds=_bounds)

    # assemble data for surrogate model training  
    comps_all = test_function.sim.comps 
    spectra_all = test_function.sim.spectra 
    print('Data shapes : ', comps_all.shape, spectra_all.shape)

    # obtain new set of compositions to synthesize
    new_x, gp_model, acquisition, train_x = run_iteration(comps_all, spectra_all)
    np.save(SAVE_DIR+'comps_%d.npy'%(i+1), new_x.cpu().numpy()) 

    # visualize models trained so far
    expt_sim.plot(PLOT_DIR+'train_spectra_%d.png'%i)
    plot_iteration(i, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
    plt.savefig(PLOT_DIR+'itr_%d.png'%i)
    plt.close()
    plot_gpmodel_expt(test_function, gp_model, np_model, PLOT_DIR+'gpmodel_itr_%d.png'%i)
    plot_phasemap_pred(test_function, gp_model, np_model, PLOT_DIR+'compare_spectra_pred_%d.png'%i) 