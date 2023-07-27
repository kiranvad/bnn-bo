import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from models import SingleTaskGP, MultiTaskGP, SingleTaskDKL, MultiTaskDKL

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
    sampler = StochasticSampler(sample_shape=torch.Size([1024]))
    if test_function.num_objectives==1:
        acqf = qUpperConfidenceBound(model=model, beta=100, sampler=sampler)
    else:
        dim = train_y.shape[1]
        weights = torch.ones(dim)/dim
        posterior_transform = ScalarizedPosteriorTransform(weights.to(train_x))
        acqf = qUpperConfidenceBound(model=model, 
        beta=100, 
        sampler=sampler,
        posterior_transform = posterior_transform
        )

    return acqf