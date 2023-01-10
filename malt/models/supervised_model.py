"""Wrapper around regressor and representation to hold the entire model. """
# =============================================================================
# IMPORTS
# =============================================================================
import functools
import abc
import torch
import gpytorch
import pyro
from typing import Any
from .regressor import Regressor
from .representation import Representation
from .utils import to_pyro

# =============================================================================
# BASE CLASSES
# =============================================================================
class SupervisedModel(torch.nn.Module):
    """A supervised model.

    Parameters
    ----------
    representation : Representation
        Module to project small molecule graph to latent embeddings.

    regressor : Regressor
        Module to convert latent embeddings to likelihood parameters.

    likelihood : Likelihood
        Module to convert likelihood parameters and data to probabilities.

    Methods
    -------
    condition

    loss

    """

    def __init__(
        self,
        representation: Representation,
        regressor: Regressor,
    ) -> None:
        super().__init__()

        assert representation.out_features == regressor.in_features
        self.representation = representation
        self.regressor = regressor

    def forward(self, x):
        """ Make predictive posterior. """
        representation = self.representation(x)
        posterior = self.regressor(representation)
        return posterior

    def loss(self, x, y):
        """Default loss function. """
        representation = self.representation(x)
        loss = self.regressor.loss(representation, y)
        return loss

    def pyro_guide(self, x, y=None):
        guide = pyro.infer.autoguide.AutoGuideList(self.pyro_model)
        guide.append(self.representation.get_pyro_guide())
        guide.append(self.regressor.get_pyro_guide())
        return guide(x, y)

    def pyro_model(self, x, y=None):
        representation = self.representation.pyro_model(x)
        posterior = self.regressor.pyro_model(representation)
        return posterior
