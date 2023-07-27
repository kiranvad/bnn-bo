import numpy as np 
import torch
import matplotlib.pyplot as plt
import pdb
from botorch.test_functions import SyntheticTestFunction

# create synthetic data
class PhaseMappingTestFunction(SyntheticTestFunction):
    r"""Base class for phasemapping test functions

    """
    _check_grad_at_opt: bool = False

    def __init__(self, sim, np_model, dim = 2, num_objectives=3):
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.sim = sim 
        self.dim = dim
        bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=0.0, negate=False, bounds=bounds)
        self.np_model = np_model
        self.num_objectives = num_objectives

    def evaluate_true(self, X):
        spectra = torch.zeros((X.shape[0], self.sim.n_domain)).to(X)
        for i, xi in enumerate(X):
            si = self.sim.simulate(xi.cpu().numpy())
            spectra[i] = torch.tensor(si, dtype=torch.float32).to(X)
        t = torch.from_numpy(self.sim.t.astype(np.float32))
        t = t.repeat(X.shape[0], 1).to(X)
        with torch.no_grad():
            z, _ = self.np_model.xy_to_mu_sigma(t.unsqueeze(2), spectra.unsqueeze(2))

        return z   

