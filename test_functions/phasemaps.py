import numpy as np 
import torch
import matplotlib.pyplot as plt
import pdb

# create synthetic data
class PrabolicPhases:
    def __init__(self, n_grid=50, n_domain=100, use_random_warping=False, noise=False):
        """ Simulate a phasemap with domain warping of functions
        """
        self.n_domain = n_domain
        self.noise = noise
        self.t = np.linspace(0,1, num=self.n_domain)
        self.n_grid = n_grid
        x = np.linspace(0,1, n_grid)
        y = np.linspace(0,1, n_grid)
        X,Y = np.meshgrid(x,y)
        self.points = np.vstack([X.ravel(), Y.ravel()]).T
        self.phase1 = lambda x : 0.5*(x)**2+0.45
        self.phase2 = lambda x : -0.45*(x)**2+0.55
        self.use_random_warping = use_random_warping
        
    def g(self, t, p):
        out = np.ones(self.t.shape)
        for i in range(1,p+1):
            zi = np.random.normal(1, 0.1)
            mean = (2*i-1)/(2*p)
            std = 1/(3*p)
            out += zi*self.phi(t, mean, std)

        return out
    
    def phi(self, t, mu, sigma):
        factor = 1/(2*(sigma**2))
        return np.exp(-factor*(t-mu)**2)
    
    def gamma(self):
        if not self.use_random_warping:
            
            return self.t

        a = np.random.uniform(-3, 3)
        if a==0:
            gam = self.t
        else:
            gam = (np.exp(a*self.t)-1)/(np.exp(a)-1)

        return gam

    def simulate(self, c):
        label = self.get_label(c)
        y = self.g(self.gamma(), label)

        if self.noise:
            y += 0.05*np.random.normal(size=self.n_domain)

        return y
    
    def get_label(self, c):
        if c[1]-self.phase1(c[0])>0:
            label = 1
        elif c[1]-self.phase2(c[0])<0:
            label = 2
        else:
            label = 3
            
        return label

    def generate(self):
        self.labels = [self.get_label(ci) for ci in self.points]            
        self.F = [self.simulate(ci) for ci in self.points]

        return

    def plot(self, fname):
        fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
        axs = axs.T
        c = np.linspace(0, 1, 10)
        for i in range(10):
            for j in range(10):
                cij = np.array([c[i], c[j]])
                axs[i,9-j].plot(self.t, self.simulate(cij))
                axs[i,9-j].set_xlim(0, 1)
                axs[i, 9-j].axis('off')
        fig.supxlabel('C1', fontsize=20)
        fig.supylabel('C2', fontsize=20) 
        plt.savefig(fname)
        plt.close()


class GaussianPhases(PrabolicPhases):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_label(self, c):
        pos = self._rescale_pos(c)
        Z1 = self._multivariate_gaussian(pos, 
                                np.array([-1, -1]), 
                                np.array([[ 0.8, 0.01], [0.01, 0.8]])
                                )
        Z2 = self._multivariate_gaussian(pos, 
                                np.array([2., 2.]), 
                                np.array([[ 0.2 , 0.01], [0.01, 0.2]])
                                )
        Z3 = self._multivariate_gaussian(pos, 
                                np.array([-2., 2.5]), 
                                np.array([[ 0.4 , 0.01], [0.01, 0.4]])
                                )
        probs = np.asarray([Z1, Z2, Z3])
        argmax = np.argmax(probs)
        if probs[argmax]>1e-2:
            return argmax+1 
        else:
            return 0

    def simulate(self, c):
        label = self.get_label(c)
        if label==0:
            y = self.phi(self.t, 1e-3, 1e-1)
        else:
            y = self.g(self.gamma(), label)

        if self.noise:
            y += 0.05*np.random.normal(size=self.n_domain)

        return y   

    def _multivariate_gaussian(self, x, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = ((x-mu).T)@Sigma_inv@(x-mu)

        return np.exp(-fac / 2) / N

    def _rescale_pos(self, pos):
        """Give a pos, rescale it to (-3,3)"""
        x, y = pos 
        x_ = x*6-3
        y_ = y*6-3

        return x_, y_