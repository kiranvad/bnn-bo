import argparse
import json
import os
import sys
import time
import traceback
import shutil
from datetime import datetime
import pdb
import numpy as np

import torch
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from models import SingleTaskGP, MultiTaskGP, SingleTaskDKL, MultiTaskDKL
from test_functions import PhaseMappingTestFunction
from utils import *
from activephasemap.activelearn.simulators import PrabolicPhases, GaussianPhases, GNPPhases
from activephasemap.np.neural_process import NeuralProcess

BATCH_SIZE = 4
N_INIT_POINTS = 4
N_ITERATIONS = 3
RANDOM_SEED = 2158
MODEL_NAME = "gp"
SIMULATOR = "parabolic"
SAVE_DIR = './results/phasemaps/%s_%s/'%(SIMULATOR, MODEL_NAME)
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
torch.manual_seed(RANDOM_SEED)

if SIMULATOR=="parabolic":
    sim = PrabolicPhases(n_grid=100, use_random_warping=False, noise=True)
elif SIMULATOR=="gaussians":
    sim = GaussianPhases(n_grid=100, use_random_warping=False, noise=True)
elif SIMULATOR=="goldnano":
    dirloc = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/gold_nano_grid/"
    sim = GNPPhases(dirloc)

sim.generate()
sim.plot(SAVE_DIR+'phasemap.png')

# Specify the Neural Process model
if SIMULATOR=="goldnano":
    N_LATENT = 2
    PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/results_pretrain/trained_model.pt"
else:
    N_LATENT = 3
    PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/phasemaps/pretrain/trained_model.pt"

np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

test_function = PhaseMappingTestFunction(sim=sim)
input_dim = test_function.dim
output_dim = N_LATENT 

init_x = initialize_points(test_function.bounds, N_INIT_POINTS, output_dim, device)
init_y = test_function(np_model, init_x)
bounds = test_function.bounds.to(device)

standard_bounds = torch.ones(2, test_function.dim).to(device)
standard_bounds[0] = 1e-5

train_x = init_x
train_y = init_y

if MODEL_NAME=="gp":
    model_args = {"model":"gp"}
elif MODEL_NAME=="dkl":
    model_args = {"model": "dkl",
    "regnet_dims": [32,32,32],
    "regnet_activation": "tanh",
    "pretrain_steps": 1000,
    "train_steps": 1000
    }


t = time.time()
for i in range(N_ITERATIONS):
    print("\niteration %d" % i)
    gp_model = initialize_model(MODEL_NAME, model_args, input_dim, output_dim, device)

    # fit model on normalized x
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
    # evaluate new y values and save
    new_y = test_function(np_model, new_x)

    if np.remainder(100*(i)/N_ITERATIONS,10)==0:
        plot_iteration(i, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)
        plt.close()
        plot_gpmodel(test_function, gp_model, np_model, SAVE_DIR+'gpmodel_itr_%d.png'%i)   

    del acqf_values
    del normalized_candidates
    torch.cuda.empty_cache()

    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

"""Plotting after training"""
plot_iteration(i, test_function, train_x, gp_model, np_model, acquisition, N_LATENT)
plt.savefig(SAVE_DIR+'itr_%d.png'%i) 
plot_phasemap_pred(test_function, gp_model, np_model, SAVE_DIR+'compare_spectra_pred.png')
plot_gpmodel(test_function, gp_model, np_model, SAVE_DIR+'model_c2z.png')   

fig, ax = plt.subplots(figsize=(10,10))
plot_gpmodel_grid(ax, test_function, gp_model, np_model, show_sigma=False)
plt.savefig(SAVE_DIR+'phasemap_pred.png') 

torch.save(train_x.cpu(), "%s/train_x.pt" % SAVE_DIR)
torch.save(train_y.cpu(), "%s/train_y.pt" % SAVE_DIR)
torch.save(gp_model.state_dict(), SAVE_DIR+'model.pt')
