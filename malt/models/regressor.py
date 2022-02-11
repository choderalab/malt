# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch

# =============================================================================
# BASE CLASSES
# =============================================================================
class Regressor(torch.nn.Module, abc.ABC):
    """Base class for a regressor."""

    def __init__(self, in_features, out_features, *args, **kwargs):
        super(Regressor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


# =============================================================================
# KERNELS
# =============================================================================
class RBF(torch.nn.Module):
    r"""A Gaussian Process Kernel that hosts parameters.

    Note
    ----
    l could be either of shape 1 or hidden dim

    """

    def __init__(self, in_features, scale=0.0, variance=0.0, ard=True):

        super(RBF, self).__init__()

        if ard is True:
            self.scale = torch.nn.Parameter(scale * torch.ones(in_features))

        else:
            self.scale = torch.nn.Parameter(torch.tensor(scale))

        self.variance = torch.nn.Parameter(torch.tensor(variance))

    def distance(self, x, x_):
        """ Distance between data points. """
        return torch.norm(x[:, None, :] - x_[None, :, :], p=2, dim=2)

    def forward(self, x, x_=None):
        """ Forward pass. """
        # replicate x if there's no x_
        if x_ is None:
            x_ = x

        # for now, only allow two dimension
        assert x.dim() == 2
        assert x_.dim() == 2

        x = x * torch.exp(self.scale)
        x_ = x_ * torch.exp(self.scale)

        # (batch_size, batch_size)
        distance = self.distance(x, x_)

        # convariant matrix
        # (batch_size, batch_size)
        k = torch.exp(self.variance) * torch.exp(-0.5 * distance)

        return k


# =============================================================================
# MODULE CLASSES
# =============================================================================
class NeuralNetworkRegressor(Regressor):
    """ Regressor with neural network. """

    def __init__(
        self,
        in_features: int = 128,
        hidden_features: int = 128,
        out_features: int = 2,
        depth: int = 2,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super(NeuralNetworkRegressor, self).__init__(
            in_features=in_features, out_features=out_features
        )
        # bookkeeping
        self.hidden_features = hidden_features
        self.out_features = out_features

        # neural network
        modules = []
        _in_features = in_features
        for idx in range(depth - 1):
            modules.append(torch.nn.Linear(_in_features, hidden_features))
            modules.append(activation)
            _in_features = hidden_features
        modules.append(torch.nn.Linear(hidden_features, out_features))

        self.ff = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.ff(x)


class ExactGaussianProcessRegressor(Regressor):
    epsilon = 1e-5

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 2,
        kernel_factory: torch.nn.Module = RBF,
        log_sigma: float = -3.0,
    ):
        assert out_features == 2
        super(ExactGaussianProcessRegressor, self).__init__(
            in_features=in_features,
            out_features=out_features,
        )

        # construct kernel
        self.kernel = kernel_factory(
            in_features=in_features,
        )

        self.in_features = in_features
        self.log_sigma = torch.nn.Parameter(
            torch.tensor(log_sigma),
        )

    def _get_kernel_and_auxiliary_variables(
        self,
        h_tr,
        y_tr,
        h_te=None,
    ):
        """ Get kernel and auxiliary variables for forward pass. """

        # compute the kernels
        k_tr_tr = self._perturb(self.kernel.forward(h_tr, h_tr))

        if h_te is not None:  # during test
            k_te_te = self._perturb(self.kernel.forward(h_te, h_te))
            k_te_tr = self._perturb(self.kernel.forward(h_te, h_tr))
            # k_tr_te = self.forward(h_tr, h_te)
            k_tr_te = k_te_tr.t()  # save time

        else:  # during train
            k_te_te = k_te_tr = k_tr_te = k_tr_tr

        # (batch_size_tr, batch_size_tr)
        k_plus_sigma = k_tr_tr + torch.exp(self.log_sigma) * torch.eye(
            k_tr_tr.shape[0], device=k_tr_tr.device
        )

        # (batch_size_tr, batch_size_tr)
        l_low = torch.linalg.cholesky(k_plus_sigma)
        l_up = l_low.t()

        # (batch_size_tr. 1)
        l_low_over_y, _ = torch.triangular_solve(
            input=y_tr, A=l_low, upper=False
        )

        # (batch_size_tr, 1)
        alpha, _ = torch.triangular_solve(
            input=l_low_over_y, A=l_up, upper=True
        )

        return k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha

    def condition(self, h_te, *args, h_tr=None, y_tr=None, **kwargs):
        r"""Calculate the predictive distribution given `h_te`.

        Note
        ----
        Here we allow the specification of sampler but won't actually
        use it here in this version.

        Parameters
        ----------
        h_te : `torch.Tensor`, `shape=(n_te, hidden_dimension)`
            Test input.

        h_tr : `torch.Tensor`, `shape=(n_tr, hidden_dimension)`
             (Default value = None)
             Training input.

        y_tr : `torch.Tensor`, `shape=(n_tr, 1)`
             (Default value = None)
             Test input.

        sampler : `torch.optim.Optimizer` or `pinot.Sampler`
             (Default value = None)
             Sampler.

        Returns
        -------
        distribution : `torch.distributions.Distribution`
            Predictive distribution.

        """

        # get parameters
        (
            k_tr_tr,
            k_te_te,
            k_te_tr,
            k_tr_te,
            l_low,
            alpha,
        ) = self._get_kernel_and_auxiliary_variables(h_tr, y_tr, h_te)

        # compute mean
        # (batch_size_te, 1)
        mean = k_te_tr @ alpha

        # (batch_size_tr, batch_size_te)
        v, _ = torch.triangular_solve(input=k_tr_te, A=l_low, upper=False)

        # (batch_size_te, batch_size_te)
        variance = k_te_te - v.t() @ v

        # ensure symetric
        variance = 0.5 * (variance + variance.t())

        # $ p(y|X) = \int p(y|f)p(f|x) df $
        # variance += torch.exp(self.log_sigma) * torch.eye(
        #         *variance.shape,
        #         device=variance.device)

        # construct noise predictive distribution
        distribution = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                mean.flatten(), variance
            )
        )

        return distribution

    def _perturb(self, k):
        """Add small noise `epsilon` to the diagonal of covariant matrix.
        Parameters
        ----------
        k : `torch.Tensor`, `shape=(n_data_points, n_data_points)`
            Covariant matrix.
        Returns
        -------
        k : `torch.Tensor`, `shape=(n_data_points, n_data_points)`
            Perturbed covariant matrix.
        """
        # introduce noise along the diagonal
        noise = self.epsilon * torch.eye(*k.shape, device=k.device)

        return k + noise

    def loss(self, h_tr, y_tr, *args, **kwargs):
        r"""Compute the loss.
        Note
        ----
        Defined to be negative Gaussian likelihood.
        Parameters
        ----------
        h_tr : `torch.Tensor`, `shape=(n_training_data, hidden_dimension)`
            Input of training data.
        y_tr : `torch.Tensor`, `shape=(n_training_data, 1)`
            Target of training data.
        Returns
        -------
        nll : `torch.Tensor`, `shape=(,)`
            Negative log likelihood.
        """
        # point data to object
        self._h_tr = h_tr
        self._y_tr = y_tr

        # get the parameters
        (
            k_tr_tr,
            k_te_te,
            k_te_tr,
            k_tr_te,
            l_low,
            alpha,
        ) = self._get_kernel_and_auxiliary_variables(h_tr, y_tr)

        import math

        # we return the exact nll with constant
        nll = (
            0.5 * (y_tr.t() @ alpha)
            + torch.trace(l_low)
            + 0.5 * y_tr.shape[0] * math.log(2.0 * math.pi)
        )

        return nll



class BiophysicalRegressor(Regressor):
    r""" Biophysically inspired model

    Parameters
    ----------

    log_sigma : `float`
        ..math:: \log\sigma observation noise

    base_regressor : a regressor object that generates a latent F

    """    
    def __init__(
        self,
        base_regressor: torch.nn.Module = ExactGaussianProcessRegressor,
        *args,
        **kwargs
    ):
        super(BiophysicalRegressor, self).__init__(
            in_features=base_regressor.in_features,
            out_features=base_regressor.out_features
        )
        self.base_regressor = base_regressor
        self.log_sigma_measurement = torch.nn.Parameter(torch.zeros(1))

    def _hill_langmuir(self, delta_g_samples, concentration=1e-3):
        r"""
        Note
        ----
        ..math: \theta = 1 / ( 1 + (\frac{K_A}{[L]})^n )

        Transform samples of `delta_g` using
            f(dG) = 1.0 / (1.0 + \exp(delta_g) / concentration)

        Assumes Hill constant n = 1 (noncooperative binding).

        Parameters
        ----------
        delta_g_samples : `torch.Tensor`, `shape=(n_samples, n_training_data)`
            Samples from posterior predictive distribution from base regressor of delta Gibbs energy.
        concentration : `float` or `torch.Tensor`, `shape=(n_concentrations,)`
            Range of molar concentrations to predict on given delta G.

        Returns
        -------
        Transformed samples from posterior predictive, per concentrations provided.
        
        """
        return 1 / (1 + torch.exp(delta_g_samples[:, :, None]) / concentration)


    def condition(self, h=None, concentration=1e-3, num_samples=100, *args, **kwargs):
        r"""Calculate the predictive distribution for provided concentrations given `h_te`.

        Note
        ----
        Transforms samples of delta G distribution using function

            ..math:: f(dG) = 1.0 / (1.0 + \exp(-delta_g) / concentration)

        Parameters
        ----------
        h : `torch.Tensor`, `shape=(n_te, hidden_dimension)`
            Continuous hidden representation of inputs.

        concentration : `float` or `torch.Tensor`, `shape=(n_concentrations,)`
            Select molar concentrations to predict for given inputs.

        Returns
        -------
        distribution : `torch.distributions.Distribution`
            Predictive distribution.

        """
        if isinstance(concentration, torch.Tensor):
            assert h.shape[0] == concentration.shape[0]
        
        # we will treat this distribution as delta Gibbs free energy
        delta_g = self.base_regressor.condition(h, *args, **kwargs)

        print(delta_g)

        # sample distribution of delta G
        delta_g_samples = delta_g.rsample((num_samples,))

        # function of concentration given delta G samples
        f_c_given_delta_g_samples = self._hill_langmuir(
            delta_g_samples, concentration = concentration,
        )

        # create normal distribution
        f_c_given_delta_g = torch.distributions.normal.Normal(
            loc = f_c_given_delta_g_samples.mean(axis=0),
            scale = torch.exp(self.log_sigma_measurement)
        )

        return f_c_given_delta_g


    def loss(self, h_tr=None, y_tr=None, concentration=None, *args, **kwargs):
        
        # if isinstance(self.base_regressor, ExactGaussianProcessRegressor):
        #     # point data to object
        #     self.base_regressor._h_tr = h_tr
        #     self.base_regressor._y_tr = y_tr

        # import pdb; pdb.set_trace()
        dummy_y = torch.ones((h_tr.shape[0], 1), device='cuda:0')

        f_c_given_delta_g = self.condition(
            h=h_tr,
            h_tr=h_tr,
            y_tr=dummy_y,
            concentration=concentration,
            *args,
            **kwargs
        )
        
        # compute negative log likelihood for all measurements
        nll = -f_c_given_delta_g.log_prob(y_tr)

        # only sum loss for real measurements
        loss = nll[y_tr != 0.0].sum()
        # import pdb; pdb.set_trace()
        
        return loss