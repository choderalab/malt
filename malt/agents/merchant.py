import abc
from malt.data.dataset import Dataset
from .agent import Agent

class Merchant(Agent):
    """ Base class for all merchants.

    Methods
    -------
    catalogue (*args, **kwargs):
        Return an iterator over the catalogue of the merchant.

    order (*args, **kwargs):
        Place an order from the merchant.

    """

    def __init__(self):
        super(Merchant, self).__init__()

    @abc.abstractmethod
    def catalogue(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def order(self, *args, **kwargs):
        raise NotImplementedError

class DatasetMerchant(Merchant):
    """ Merchant with a candidate pool.

    Parameters
    ----------
    dataset : Dataset
        A dataset of Points.
    """
    def __init__(
        self,
        dataset: Dataset,
    ):
        super(DatasetMerchant, self).__init__()
        self.dataset = dataset.clone().erase_annotation()

    def catalogue(self):
        return self.dataset

    def order(self, dataset):
        """ Order molecules in subset.

        Parameters
        ----------
        dataset : malt.Dataset
            A
        """

        self.dataset -= dataset
        return dataset