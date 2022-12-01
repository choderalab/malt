from typing import Any
import functools
import torch
import pyro

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def _rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)

def _wrap_pyro_model(model):
    class WrappedModel(pyro.nn.PyroModule):
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, x, y=None):
            distribution = self._model(x)
            distribution = torch.distributions.Independent(distribution, 1)
            with pyro.plate("observed_data"):
                pyro.sample("obs", distribution, obs=y.unsqueeze(-1))
            return distribution

    return WrappedModel(model)

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def to_pyro(
        model: Any,
        guide: pyro.infer.autoguide.AutoGuide=pyro.infer.autoguide.AutoDelta,
    ) -> pyro.nn.PyroModule:
    """Convert a PyTorch module to pyro module.

    Parameters
    ----------
    model : SupervisedModel
        Module to be converted to pyro.

    Returns
    -------
    pyro.PyroModule
        Pyro module.


    """
    # convert model to pyro_model
    pyro.nn.module.to_pyro_module_(model)

    if guide != pyro.infer.autoguide.AutoDelta:
        for name, parameter in model.named_parameters():
            if "weight" in name:
                setattr(
                    model,
                    name,
                    pyro.nn.PyroSample(
                        pyro.distributions.Normal(0, 1)
                        .expand(parameter.shape)
                        .to_event(parameter.shape[-1]),
                    )
                )


    model = _wrap_pyro_model(model)
    guide = guide(model)

    return model, guide
