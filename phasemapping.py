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
from test_functions import PrabolicPhases
from utils import initialize_model, initialize_points, construct_acqf_by_model 

sys.path.append('/mmfs1/home/kiranvad/kiranvad/neural-processes') 
from neural_process import NeuralProcess

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import Normalize

BATCH_SIZE = 2
N_INIT_POINTS = 2
N_ITERATIONS = 10
RANDOM_SEED = 2158
N_LATENT = 3
SAVE_DIR = './results/sinx/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

sim = PrabolicPhases()
input_dim = 2
output_dim = N_LATENT 
bounds = [(0.0, 1.0) for _ in range(input_dim)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.manual_seed(RANDOM_SEED)

# define composite model 

# Specify the Neural Process model
np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load('../models/gaussians.pt', map_location=device))

init_x, init_y = initialize_points(sim, N_INIT_POINTS, output_dim, device)
bounds = torch.tensor(bounds).to(init_x)

standard_bounds = torch.zeros(2, test_function.dim).to(init_x)
standard_bounds[1] = 1

train_x = init_x
train_y = init_y

model_name = "gp"
model_args = {"model":"gp"}
model = initialize_model(model_name, model_args, input_dim, output_dim, device)

t = time.time()
for i in range(N_ITERATIONS):
    print("\niteration %d" % i)

    # fit model on normalized x
    model_start = time.time()
    normalized_x = normalize(train_x, bounds).to(train_x)
    model.fit_and_save(normalized_x, train_y, SAVE_DIR)
    model_end = time.time()
    print("fit time", model_end - model_start)
    
    acq_start = time.time()
    acquisition = construct_acqf_by_model(model, normalized_x, train_y, test_function)
    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, standard_bounds, q=BATCH_SIZE, num_restarts=2, raw_samples=16, return_best_only=False,
        options={"batch_limit": 1, "maxiter": 10})

    # calculate acquisition values after rounding
    acqf_values = acquisition(normalized_candidates)
    acq_end = time.time()
    print("acquisition time", acq_end - acq_start)

    best_index = acqf_values.max(dim=0).indices.item()
    candidates = unnormalize(normalized_candidates.detach(), bounds=bounds)
    new_x = candidates[best_index].to(train_x)
    # evaluate new y values and save
    new_y = test_function(new_x)

    plot_iteration(test_function, train_x, train_y, new_x, new_y, model, acquisition)
    plt.savefig(SAVE_DIR+'/Itr_%d.png'%i, dpi=600)

    del acquisition
    del acqf_values
    del normalized_candidates
    torch.cuda.empty_cache()

    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    
torch.save(train_x.cpu(), "%s/train_x.pt" % SAVE_DIR)
torch.save(train_y.cpu(), "%s/train_y.pt" % SAVE_DIR)
torch.save(model.state_dict(), SAVE_DIR+'model.pt')
