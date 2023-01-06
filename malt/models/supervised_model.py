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

    def guide(self, x, y=None):
        x = self.representation(x)
        posterior = self.regressor.guide(x)
        return posterior

    def model(self, x, y=None):
        representation = self.representation.model(x)
        posterior = self.regressor.model(representation)
        return posterior
