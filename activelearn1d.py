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
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.test_functions import Ackley
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from models import SingleTaskGP, MultiTaskGP, SingleTaskDKL, MultiTaskDKL
from test_functions import SinX 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import Normalize
# define some utility functions

def initialize_model(model_name, model_args, input_dim, output_dim, device):
    if model_name == 'gp':
        if output_dim == 1:
            return SingleTaskGP(model_args, input_dim, output_dim)
        else:
            return MultiTaskGP(model_args, input_dim, output_dim)
    elif model_name == 'dkl':
        if output_dim == 1:
            return SingleTaskDKL(model_args, input_dim, output_dim, device)
        else:
            return MultiTaskDKL(model_args, input_dim, output_dim, device)
    else:
        raise NotImplementedError("Model type %s does not exist" % model_name)


def initialize_points(test_function, n_init_points, output_dim, device):
    if n_init_points < 1:
        init_x = torch.zeros(1, 1).to(device)
    else:
        bounds = test_function.bounds.to(device, dtype=torch.float64)
        init_x = draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)
    init_y = test_function(init_x)

    return init_x, init_y


def construct_acqf_by_model(model, train_x, train_y, test_function):
    sampler = StochasticSampler(sample_shape=torch.Size([128]))
    qNIPV = qNegIntegratedPosteriorVariance(model=model, 
    mc_points=mc_points, 
    sampler=sampler)

    return qNIPV

def plot_iteration(test_function, train_x, train_y, new_x, new_y, model, acquisition):
    with torch.no_grad():
        num_grid_spacing = 100
        bounds = test_function.bounds.squeeze()
        points_t = torch.linspace(bounds[0], bounds[1], num_grid_spacing).to(train_x)
        x = points_t.cpu().numpy().squeeze()
        posterior = model.posterior(points_t)
        posterior_mean = posterior.mean.cpu().numpy()
        lower, upper = posterior.mvn.confidence_region()
        confidence = np.abs((lower-upper).cpu().numpy())        
        acq_values = acquisition(points_t.reshape(num_grid_spacing, 1, 1)).cpu().numpy()

        # plot surrogate model
        fig, ax = plt.subplots()
        ax.plot(x, posterior_mean)
        ax.fill_between(x, lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)

        # plot acquisiotn function
        ax.plot(x, acq_values)

        # plot data collected
        ax.scatter(train_x.cpu().numpy(), train_y.cpu().numpy(), marker='x', color='k')
        ax.scatter(new_x.cpu().numpy(), new_y.cpu().numpy(), marker='x', color='r')    


    return


BATCH_SIZE = 2
N_INIT_POINTS = 2
N_ITERATIONS = 5
SAVE_DIR = './results/sinx/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

args = json.load(open("./config/ackley.json", 'r'))

test_function = SinX(dim=1, noise_std=0.2)
input_dim = test_function.dim
output_dim = test_function.num_objectives

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.manual_seed(int(args["seed"]))

init_x, init_y = initialize_points(test_function, N_INIT_POINTS, output_dim, device)
bounds = test_function.bounds.to(init_x)

standard_bounds = torch.zeros(2, test_function.dim).to(init_x)
standard_bounds[1] = 1

train_x = init_x
train_y = init_y

model_dict = args["models"] 
model_name = "gp"
model_args = model_dict[model_name]
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
    acquisition = construct_acqf_by_model(model_name, model, normalized_x, train_y, test_function)
    normalized_candidates, acqf_values = optimize_acqf(
        acquisition, standard_bounds, q=BATCH_SIZE, num_restarts=2, raw_samples=16, return_best_only=False,
        options={"batch_limit": 1, "maxiter": 10})
    candidates = unnormalize(normalized_candidates.detach(), bounds=bounds)

    # calculate acquisition values after rounding
    acqf_values = acquisition(normalized_candidates)
    acq_end = time.time()
    print("acquisition time", acq_end - acq_start)

    best_index = acqf_values.max(dim=0).indices.item()
    # best x is best acquisition value after rounding
    new_x = candidates[best_index].to(train_x)
    # evaluate new y values and save
    new_y = test_function(new_x)

    plot_iteration(test_function, train_x, train_y, new_x, new_y, model, acquisition)
    plt.savefig(SAVE_DIR+'/Itr_%d.png'%i)

    del acquisition
    del acqf_values
    del normalized_candidates
    torch.cuda.empty_cache()

    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    
torch.save(train_x.cpu(), "%s/train_x.pt" % SAVE_DIR)
torch.save(train_y.cpu(), "%s/train_y.pt" % SAVE_DIR)
torch.save(model.state_dict(), SAVE_DIR+'model.pt')
