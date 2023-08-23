import numpy as np 
import torch
import matplotlib.pyplot as plt
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create synthetic data
class PhaseMappingTestFunction:
    r"""Base class for phasemapping test functions

    """

    def __init__(self, sim, dim = 2, num_objectives=3):
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.sim = sim 
        self.dim = dim
        self._bounds = [(1e-4, 1.0) for _ in range(self.dim)]
        self.bounds = torch.tensor(self._bounds).transpose(-1, -2).to(device)
        self.num_objectives = num_objectives

    def evaluate_true(self, np_model, X):
        spectra = torch.zeros((X.shape[0], self.sim.n_domain)).to(device)
        for i, xi in enumerate(X):
            si = self.sim.simulate(xi.cpu().numpy())
            spectra[i] = torch.tensor(si).to(device)
        t = torch.from_numpy(self.sim.t)
        t = t.repeat(X.shape[0], 1).to(device)
        with torch.no_grad():
            z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2), spectra.unsqueeze(2))

        return z  

    __call__ = evaluate_true 
