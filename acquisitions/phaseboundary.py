import torch
from autophasemap import compute_elastic_kmeans, BaseDataSet, compute_BIC
from autophasemap.geometry import SquareRootSlopeFramework, WarpingManifold
from autophasemap.diffusion import DiffusionMaps
from .utils import MinMaxScaler, get_twod_grid
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoPhaseMapDataSet(BaseDataSet):
    def __init__(self, C, q, Iq, n_domain = 100):
        super().__init__(n_domain=n_domain)
        self.t = np.linspace(0,1, num=self.n_domain)
        self.q = q
        self.N = C.shape[0]
        self.Iq = Iq
        self.C = C
    
    def generate(self, process=None):
        if process=="normalize":
            self.F = [self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]) for i in range(self.N)]
        elif process=="smoothen":
            self.F = [self._smoothen(self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]), window_length=7, polyorder=3) for i in range(self.N)]
        elif process is None:
            self.F = [self.Iq[i,:] for i in range(self.N)]
            
        return


class PhaseBoundaryPenalty(torch.nn.Module):
    r"""A penalty funcion based on phase boundaries to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, test_function, gp_model, np_model, min_clusters=2, max_clusters=8):
        super().__init__()
        self.test_function = test_function
        self.gp_model = gp_model
        self.np_model = np_model
        bounds = self.test_function.bounds.cpu().numpy()
        grid_comps = get_twod_grid(10, bounds = bounds)
        scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
        scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
        n_grid_samples = grid_comps.shape[0]
        n_spectra_dim =  self.test_function.sim.t.shape[0]
        grid_spectra = np.zeros((n_grid_samples, n_spectra_dim))
        with torch.no_grad():
            for i in range(n_grid_samples):
                mu, _ = from_comp_to_spectrum(self.test_function, self.gp_model, self.np_model, 
                grid_comps[i,:].reshape(1, 2))
                grid_spectra[i,:] = mu.cpu().squeeze().numpy()

        self.data = AutoPhaseMapDataSet(grid_comps, test_function.sim.t, grid_spectra, n_domain=n_spectra_dim)
        self.data.generate()
        sweep_n_clusters = np.arange(min_clusters,max_clusters)
        BIC = []
        for n_clusters in sweep_n_clusters:
            out = compute_elastic_kmeans(self.data, n_clusters, max_iter=50, verbose=0, smoothen=False)
            BIC.append(compute_BIC(self.data, out.fik_gam, out.qik_gam, out.delta_n))

        self.min_bic_clusters = sweep_n_clusters[np.argmin(BIC)]
        self.out = compute_elastic_kmeans(self.data, min_bic_clusters, max_iter=100, verbose=0, smoothen=True)

    def forward(self, X):
        SRSF = SquareRootSlopeFramework(self.data.t)
        diffmap = DiffusionMaps(self.data.C)
        mu, _ = from_comp_to_spectrum(self.test_function, self.gp_model, self.np_model, X.reshape(1, 2))
        F = mu.cpu().squeeze().numpy()
        q = SRSF.to_srsf(F)
        dists = np.zeros(self.min_bic_clusters)
        for k in range(self.min_bic_clusters):
            gamma = SRSF.get_gamma(out.templates[k], q)
            f_gam = SRSF.warp_f_gamma(F, gam)
            q_gam = SRSF.to_srsf(f_gam)
            dists[k] = np.sqrt(np.trapz((out.templates[k] - q_gam)**2, data.t))
        
        s_norm, s_hat, d_smoothened = diffmap.get_asymptotic_function(dists)
        entropy = [-d*np.log(d) for d in d_smoothened]
        
        return torch.from_numpy(np.sum(np.asarray(entropy).T, axis=1)).to(device)