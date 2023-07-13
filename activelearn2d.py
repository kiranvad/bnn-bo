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
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.test_functions import Ackley
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from models import SingleTaskGP, MultiTaskGP, SingleTaskDKL, MultiTaskDKL

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import Normalize
# define some utility functions

def round(test_function_name, x):
    if test_function_name == "oil":
        x[..., 2:] = torch.floor(x[..., 2:])
    elif test_function_name == "cco":
        x[..., 15:] = torch.floor(x[..., 15:])
    elif test_function_name == "pest":
        x = torch.floor(x)
    return x

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


def initialize_points(test_function, n_init_points, output_dim, device, test_function_name):
    if n_init_points < 1:
        init_x = torch.zeros(1, 1).to(device)
    else:
        bounds = test_function.bounds.to(device, dtype=torch.float64)
        init_x = draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)
        init_x = round(test_function_name, init_x)
    init_y = test_function(init_x)
    # add explicit output dimension
    if output_dim == 1:
        init_y = init_y.unsqueeze(-1)
    return init_x, init_y


def construct_acqf_by_model(model_name, model, train_x, train_y, test_function):
    sampler = StochasticSampler(sample_shape=torch.Size([128]))
    if test_function.num_objectives == 1:
        qEI = qExpectedImprovement(
            model=model,
            best_f=train_y.max(),
            sampler=sampler
        )
        return qEI
    else: # multi-objective
        with torch.no_grad():
            pred = model.posterior(train_x).mean
            pred = pred.squeeze(-1) # TODO: Laplace?
        partitioning = FastNondominatedPartitioning(
            ref_point=test_function.ref_point.to(train_x),
            Y=pred,
        )
        qEHVI = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=test_function.ref_point.to(train_x),
            partitioning=partitioning,
            sampler=sampler
        )
        return qEHVI

def get_test_function(test_function, seed):
    test_function = test_function.lower()
    if "ackley" in test_function:
        if test_function == "ackley":
            dim = 2
        else:
            dim = int(test_function.split('_')[-1])
        return Ackley(dim=dim, negate=True)

def plot_iteration(test_function, train_x, new_x, model, acquisition):
    with torch.no_grad():
        num_grid_spacing = 20
        x = np.linspace(*test_function.bounds[:,0].cpu().numpy(), num=num_grid_spacing)
        y = np.linspace(*test_function.bounds[:,1].cpu().numpy(), num=num_grid_spacing)
        XX, YY = np.meshgrid(x,y)
        points = np.vstack([XX.ravel(), YY.ravel()]).T
        points_t = torch.tensor(points).to(train_x)
        posterior = model.posterior(points_t)
        posterior_mean = posterior.mean.cpu().numpy()
        Z = posterior_mean.reshape(num_grid_spacing,num_grid_spacing)
        acq_values = acquisition(points_t.reshape(400,1,2)).cpu().numpy()

    # plot surrogate model
    fig, axs = plt.subplots(1, 2, figsize=(4*2, 4))
    im = axs[0].contourf(XX, YY, Z)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)       
    cbar = fig.colorbar(im, cax=cax)

    # plot acquisiotn function
    im = axs[1].contourf(XX, YY, acq_values.reshape(20,20), cmap=cm.coolwarm)
    axs[1].scatter(train_x[:,0].cpu().numpy(), train_x[:,1].cpu().numpy(), marker='x', color='k')
    axs[1].scatter(new_x[:,0].cpu().numpy(), new_x[:,1].cpu().numpy(), marker='*', color='k')    
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)        
    cbar = fig.colorbar(im, cax=cax)

    return


BATCH_SIZE = 4
N_INIT_POINTS = 5
N_ITERATIONS = 10
SAVE_DIR = './results/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

args = json.load(open("./config/ackley.json", 'r'))

test_function_name = 'ackley'
test_function = get_test_function(test_function_name, int(args["seed"]))
input_dim = test_function.dim
output_dim = test_function.num_objectives

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.manual_seed(int(args["seed"]))

init_x, init_y = initialize_points(test_function, N_INIT_POINTS, output_dim, device, test_function_name)
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

    plot_iteration(test_function, train_x, new_x,  model, acquisition)
    plt.savefig(SAVE_DIR+'/Itr_%d.png'%i)

    del acquisition
    del acqf_values
    del normalized_candidates
    torch.cuda.empty_cache()

    # evaluate new y values and save
    new_y = test_function(new_x)

    # add explicit output dimension
    if output_dim == 1:
        new_y = new_y.unsqueeze(-1)
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    
torch.save(train_x.cpu(), "%s/train_x.pt" % SAVE_DIR)
torch.save(train_y.cpu(), "%s/train_y.pt" % SAVE_DIR)
torch.save(model.state_dict(), SAVE_DIR+'model.pt')
