from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)

class PosteriorVariance(AnalyticAcquisitionFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(
            X=X,
            posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        variance = posterior.variance
        view_shape = (
            mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        )
        return variance.view(view_shape)

class qPosteriorVariance(MCAcquisitionFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        posterior = self.model.posterior(
            X=X,
            posterior_transform=self.posterior_transform
        )
        objective = self.objective(self.sampler(posterior), X=X)
        mu = objective.mean(dim=0)
        samples = (objective - mu).abs()
        return samples.max(dim=-1)[0].mean(dim=0)