"""Representation -> predictive posterior."""
# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
from typing import Union, Optional
import gpytorch
import pyro
from .utils import to_pyro
gpytorch.settings.debug._state = False


# =============================================================================
# BASE CLASSES
# =============================================================================
class Regressor(torch.nn.Module):
    """Base class for a regressor.

    Parameters
    ----------
    in_features : int
        Input features.

    """

    def __init__(self, in_features, *args, **kwargs):
        super().__init__()
        self.in_features = in_features

    def forward(self, representation):
        """Forward function.

        Parameters
        ----------
        representation : torch.Tensor
            Representation of the graph(s).

        Returns
        -------
        torch.distributions.Distribution
            Resutling distribution.

        """
        raise NotImplementedError

    def loss(self, representation, observation):
        """Compute the loss.

        Parameters
        ----------
        representation : torch.Tensor
            Representation of the graph(s).

        observation : torch.Tensor
            Observation associated with the graph.

        Returns
        -------
        torch.Tensor (shape=())

        """
        posterior = self.forward(representation)
        nll = -posterior.log_prob(observation.unsqueeze(-1)).mean()
        return nll

def _wrap_model(model):
    class WrappedModel(pyro.nn.PyroModule):
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, x, y=None):
            if y is not None:
                y = y.unsqueeze(-1)
            distribution = self._model(x)
            distribution = pyro.distributions.Independent(distribution, 1)
            with pyro.plate("observed_data"):
                pyro.sample("obs", distribution, obs=y)
            return distribution

    return WrappedModel(model)

class NeuralNetworkLikelihood(object):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HeteroschedasticGaussianLikelihood(NeuralNetworkLikelihood):
    """Models a heteroschedastic Gaussian likelihood to be used with
    neural network regressors. (Admits unconstrained parameters.)

    Attributes
    ----------
    in_features = 2

    Parameters
    ----------
    mu : torch.Tensor

    log_sigma : torch.Tensor

    Examples
    --------
    >>> likelihood = HeteroschedasticGaussianLikelihood()
    >>> posterior = likelihood(torch.tensor(0.0), torch.tensor(0.0))
    """
    in_features = 2
    def __call__(self, mu, log_sigma):
        return torch.distributions.Normal(
            mu, log_sigma.exp(), validate_args=False,
        )

class HomoschedasticGaussianLikelihood(NeuralNetworkLikelihood):
    """Models a homoschedastic Gaussian likelihood to be used with
    neural network regressors. (Admits unconstrained parameters.)

    Attributes
    ----------
    in_features = 2

    Parameters
    ----------
    mu : torch.Tensor

    log_sigma : torch.Tensor

    Examples
    --------
    >>> likelihood = HomoschedasticGaussianLikelihood()
    >>> posterior = likelihood(torch.tensor(0.0))
    """
    in_features = 1
    def __call__(self, mu):
        return torch.distributions.Normal(
            mu, torch.ones_like(mu), validate_args=False,
        )

class NeuralNetworkRegressor(Regressor):
    """ Regressor with neural network.

    Parameters
    ----------
    in_features : int = 128
        Input features.

    hidden_features : int = 128
        Hidden features.

    out_features : int = 2
        Output features.

    activation : torch.nn.Module = torch.nn.ELU()
        Activation function.

    likelihood : type
        Factory of likelihood function.

    """

    def __init__(
        self,
        in_features : int = 128,
        hidden_features : int = 128,
        depth : int = 2,
        activation : torch.nn.Module = torch.nn.ELU(),
        likelihood : NeuralNetworkLikelihood = \
            HeteroschedasticGaussianLikelihood(),
        *args, **kwargs,
    ):
        super(NeuralNetworkRegressor, self).__init__(
            in_features=in_features,
        )

        self._model = None
        self._guide = None

        # bookkeeping
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.likelihood = likelihood

        out_features = likelihood.in_features

        # neural network
        modules = []
        modules.append(activation)
        _in_features = in_features
        for idx in range(depth - 1):
            modules.append(torch.nn.Linear(_in_features, hidden_features))
            modules.append(activation)
            _in_features = hidden_features
        modules.append(torch.nn.Linear(hidden_features, out_features))

        self.ff = torch.nn.Sequential(*modules)

    def forward(self, x):
        parameters = self.ff(x).split(1, dim=-1)
        posterior = self.likelihood(*parameters)
        return posterior

    def get_model(self):
        if self._model is None:
            self._model = _wrap_model(to_pyro(self))
        return self._model

    def get_guide(self):
        model = self.get_model()
        guide = pyro.infer.autoguide.AutoNormal(model)
        return guide

    def model(self, x, y=None):
        model = self.get_model()
        return model(x, y)

    def guide(self, x, y=None):
        guide = self.get_guide()
        return guide(x, y)

class _ExactGaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, inputs, targets):
        super().__init__(inputs, targets, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.LinearMean(inputs.shape[-1])
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
        )

    def forward(self, x):
        mean = self.mean_module(x.tanh())
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class ExactGaussianProcessRegressor(Regressor):
    """Regressor with exact Gaussian process.

    Parameters
    ----------
    in_features : int = 128
        Input features.

    """
    initialized = False

    def __init__(
        self,
        in_features : int = 128,
        num_points: int = 0
    ):
        super().__init__(in_features=in_features)
        self.register_buffer(
            "x_placeholder", torch.zeros(num_points, in_features),
        )
        self.register_buffer(
            "y_placeholder", torch.zeros(num_points, ),
        )

        self.gp = _ExactGaussianProcess(self.x_placeholder, self.y_placeholder)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.gp,
        )

    def forward(self, representation):
        return self.gp(representation)

    def loss(self, representation, observation):
        if not self.initialized and self.training:
            self.gp.set_train_data(
                inputs=representation,
                targets=observation,
            )
            self.initialized = True

        nll = -self.mll(
            self.gp(representation),
            observation,
        ).mean()

        return nll
