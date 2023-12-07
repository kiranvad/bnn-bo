import os, sys, time, shutil, pdb
from datetime import datetime
import numpy as np

import torch
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from models import SingleTaskGP, MultiTaskGP, SingleTaskDKL, MultiTaskDKL
from test_functions import SimulatorTestFunction
from utils import *
from activephasemap.activelearn.simulators import PrabolicPhases, GaussianPhases, GNPPhases, PeptideGNPPhases
from activephasemap.np.neural_process import NeuralProcess 
from activephasemap.activelearn.surrogates import update_npmodel
from activephasemap.activelearn.pipeline import ActiveLearningDataset

BATCH_SIZE = 4
N_INIT_POINTS = 5
N_ITERATIONS = 10
MODEL_NAME = "dkl"
SIMULATOR = "peptide"
TEMPERATURE = 55 # available [15,27,35,55]
if not SIMULATOR=="peptide":
    SAVE_DIR = './results/phasemaps/%s_%s/'%(SIMULATOR, MODEL_NAME)
else:
    SAVE_DIR = './results/phasemaps/%s_%s/%d/'%(SIMULATOR, MODEL_NAME, TEMPERATURE)
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

if SIMULATOR=="parabolic":
    sim = PrabolicPhases(n_grid=100, use_random_warping=False, noise=True)
elif SIMULATOR=="gaussians":
    sim = GaussianPhases(n_grid=100, use_random_warping=False, noise=True)
elif SIMULATOR=="goldnano":
    dirloc = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/gold_nano_grid/"
    sim = GNPPhases(dirloc)
elif SIMULATOR=="peptide":
    dirloc = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/peptide_grid/%d/"%TEMPERATURE
    sim = PeptideGNPPhases(dirloc)    
    
sim.generate()

# Specify the Neural Process model
if SIMULATOR=="goldnano":
    N_LATENT = 2
    PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/results_pretrain/trained_model.pt"
    design_space_bounds = [(0.0, 7.38), (0.0,7.27)]
elif SIMULATOR=="peptide":
    N_LATENT = 2
    PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/results_pretrain/trained_model.pt"
    design_space_bounds = [(0.0, 87.0), (0.0,11.0)] 
else:
    N_LATENT = 3
    PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/phasemaps/pretrain/trained_model.pt"
    design_space_bounds = [(0.0, 1.0), (0.0,1.0)]

np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

test_function = SimulatorTestFunction(sim=sim, bounds=design_space_bounds)
input_dim = test_function.dim
output_dim = N_LATENT 

init_x = initialize_points(test_function.bounds, N_INIT_POINTS, output_dim, device)
init_y, spectra = test_function(np_model, init_x)
bounds = test_function.bounds.to(device)

_bounds = [(0.0, 1.0) for _ in range(input_dim)]
standard_bounds = torch.tensor(_bounds).transpose(-1, -2).to(device)

train_x = init_x
train_y = init_y

if MODEL_NAME=="gp":
    model_args = {"model":"gp"}
elif MODEL_NAME=="dkl":
    model_args = {"model": "dkl",
    "regnet_dims": [16,16,16],
    "regnet_activation": "tanh",
    "pretrain_steps": 0,
    "train_steps": 1000
    }


t = time.time()
data = ActiveLearningDataset(train_x,spectra)
for i in range(N_ITERATIONS):
    print("\niteration %d" % i)
    gp_model = initialize_model(MODEL_NAME, model_args, input_dim, output_dim, device)

    # fit model on normalized x
    normalized_x = normalize(train_x, bounds).to(train_x)
    gp_model.fit_and_save(normalized_x, train_y, SAVE_DIR) 
    
    acquisition = construct_acqf_by_model(gp_model, normalized_x, train_y, output_dim)
    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, 
        standard_bounds, 
        q=BATCH_SIZE, 
        num_restarts=20, 
        raw_samples=1024, 
        return_best_only=True,
        sequential=False,
        options={"batch_limit": 1, "maxiter": 10, "with_grad":True}
        )

    new_x = unnormalize(normalized_candidates.detach(), bounds=bounds)
    # evaluate new y values and save
    new_y, new_spectra = test_function(np_model, new_x)

    if np.remainder(100*(i+1)/N_ITERATIONS,10)==0:
        plot_experiment(test_function.sim.t, design_space_bounds, data)
        plt.savefig(SAVE_DIR+'train_spectra_%d.png'%i)
        # update np model with new data
        np_model, np_loss = update_npmodel(test_function.sim.t, np_model, data, num_iterations=75, verbose=False)
        plot_iteration(i, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)
        plt.close()
        plot_gpmodel(test_function, gp_model, np_model, SAVE_DIR+'gpmodel_itr_%d.png'%i)
        plot_phasemap_pred(test_function, gp_model, np_model, SAVE_DIR+'compare_spectra_pred_%d.png'%i)

    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    data.update(new_x, new_spectra)

"""Plotting after training"""
plot_experiment(test_function.sim.t, design_space_bounds, data)
plt.savefig(SAVE_DIR+'train_spectra_%d.png'%i)
plot_iteration(i, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
plt.savefig(SAVE_DIR+'itr_%d.png'%i) 
plot_phasemap_pred(test_function, gp_model, np_model, SAVE_DIR+'compare_spectra_pred_%d.png'%i)
plot_gpmodel(test_function, gp_model, np_model, SAVE_DIR+'gpmodel_itr_%d.png'%i)   

fig, ax = plt.subplots()
plot_gpmodel_grid(ax, test_function, gp_model, np_model, show_sigma=False)
plt.savefig(SAVE_DIR+'phasemap_pred.png')

torch.save(train_x.cpu(), "%s/train_x.pt" % SAVE_DIR)
torch.save(train_y.cpu(), "%s/train_y.pt" % SAVE_DIR)
torch.save(gp_model.state_dict(), SAVE_DIR+'model.pt')
