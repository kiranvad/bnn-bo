import sys, os, pdb, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from torch.utils.data import DataLoader

from activephasemap.activelearn.simulators import PhaseMappingExperiment
from activephasemap.activelearn.pipeline import ActiveLearningDataset
from activephasemap.np.neural_process import NeuralProcess
from activephasemap.np.training import NeuralProcessTrainer
from activephasemap.np.utils import context_target_split 
from activephasemap.activelearn.surrogates import NPModelDataset

SAVE_DIR = './results/pretrain/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
os.makedirs(SAVE_DIR+'/itrs/')
print('Saving the results to %s'%SAVE_DIR)

# hyper parameters
ITERATION = 3           # How many iterations data to use   
batch_size = 2          # mini-batch size to train NP model
num_context = 25        # Number of points use to train NP in a given function
num_target = 25         # Number of points used as target predictions in NP
num_iterations = 30     # Number iterations to optimize
r_dim = 50              # Dimension of representation of context points
z_dim = 2               # Dimension of sampled latent variable
h_dim = 50              # Dimension of hidden layers in encoder and decoder
learning_rate = 1e-2    # Learning rate for the Adam optimizer

PRETRAIN_LOC = "/mmfs1/home/kiranvad/kiranvad/neural-processes/examples/UV_VIS/results_pretrain/trained_model.pt"
EXPT_DIR = './experiments/gp_expt/'

neuralprocess = NeuralProcess(1, 1, r_dim, z_dim, h_dim).to(device)
neuralprocess.load_state_dict(torch.load(PRETRAIN_LOC, map_location=device))

for name, param in neuralprocess.named_parameters():
    if not 'hidden_to' in name:
        param.requires_grad = False

# Create dataset
comps, spectra = [], []
for k in range(ITERATION+1):
    comps.append(np.load(EXPT_DIR+'comps_%d.npy'%k))
    xlsx = pd.read_excel(EXPT_DIR+'spectra_%d.xlsx'%k, engine='openpyxl') 
    spectra.append(xlsx.values)
    print('Iteration %d : '%k, comps[k].shape, spectra[k].shape)
comps = np.vstack(comps)
spectra = np.vstack(spectra)

def normalize(wav, f):
    norm = np.sqrt(np.trapz(f**2, wav))

    return f/norm 

wav = np.load(EXPT_DIR+'wav.npy')
spectra_normalized = np.vstack([normalize(wav, spectra[i,:]) for i in range(len(comps))])
t = (wav - min(wav))/(max(wav) - min(wav))
data = ActiveLearningDataset(comps,spectra_normalized)
dataset = NPModelDataset(t, data.y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Visualize data samples
for i in np.random.randint(len(dataset), size=100):
    xi, yi = dataset[i]
    plt.plot(xi.cpu().numpy(), yi.cpu().numpy(), c='b', alpha=0.5)
plt.savefig(SAVE_DIR+'data_samples.png')
plt.close()

# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is
# (1, 100, 1)
x_target = torch.Tensor(np.linspace(0, 1, 100)).to(device)
x_target = x_target.unsqueeze(1).unsqueeze(0)

with torch.no_grad():
    for i in range(100):
        z_sample = torch.randn((1, z_dim)).to(device)
        mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
        plt.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0], c='b', alpha=0.5)
    plt.savefig(SAVE_DIR+'samples_before_training.png')

optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(num_context, num_context),
                                  num_extra_target_range=(num_target, num_target), 
                                  print_freq=1000)

neuralprocess.training = True
x_plot = torch.Tensor(np.linspace(0, 1, 100)).to(device)
x_plot = x_plot.unsqueeze(1).unsqueeze(0)
np_trainer.train(data_loader, num_iterations, x_plot, savedir=SAVE_DIR+'/itrs/') 

neuralprocess.training = False

with torch.no_grad():
    for i in range(100):
        z_sample = torch.randn((1, z_dim)).to(device)
        mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
        plt.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0], c='b', alpha=0.5)

    plt.savefig(SAVE_DIR+'samples_after_training.png')
    plt.close()

    # Extract a batch from data_loader
    fig, axs = plt.subplots(2,4, figsize=(4*4, 4*2))
    for ax in axs.flatten(): 
        x, y = next(iter(data_loader))
        x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],num_context, num_target)
        ax.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='tab:red')
        ax.plot(x[0:1].squeeze().cpu().numpy(), y[0:1].squeeze().cpu().numpy(), 
        color='tab:red', lw=1.0
        )
        for i in range(200):
            # Neural process returns distribution over y_target
            p_y_pred = neuralprocess(x_context, y_context, x_target)
            ax.plot(x_target.cpu().numpy()[0],
            p_y_pred.loc.cpu().numpy()[0], 
            alpha=0.05, c='b'
            )
    plt.savefig(SAVE_DIR+'samples_from_posterior.png')
    plt.close()

    # plot grid
    z1 = torch.linspace(-3,3,10)
    z2 = torch.linspace(-3,3,10)
    fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
    for i in range(10):
        for j in range(10):
            z_sample = torch.zeros((1, z_dim)).to(device)
            z_sample[0,0] = z1[i]
            z_sample[0,1] = z2[j]
            mu, sigma = neuralprocess.xz_to_y(x_target, z_sample)
            mu_, sigma_ = mu.cpu().squeeze().numpy(), sigma.cpu().squeeze().numpy()
            axs[i,j].plot(x_target.cpu().squeeze().numpy(), mu_)
            axs[i,j].fill_between(x_target.cpu().squeeze().numpy(), 
            mu_-sigma_, mu_+sigma_,alpha=0.2, color='grey'
            )
            # axs[i,j].set_title('(%.2f, %.2f)'%(z1[i], z2[j]))
    fig.supxlabel('z1')
    fig.supylabel('z2')

    plt.savefig(SAVE_DIR+'samples_in_grid.png')
    plt.close()

# %%
torch.save(neuralprocess.state_dict(), SAVE_DIR+'trained_model.pt')