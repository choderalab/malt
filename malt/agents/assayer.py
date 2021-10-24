import abc
from .agent import Agent
from malt.data.dataset import Dataset

class Assayer(Agent):
    """Assayer. Takes a point and annotates `y` field as well as (optionally)
    extra.

    Methods
    -------
    assay(dataset)
        Annotate `y` in the dataset.

    """
    def __init__(self):
        super(Assayer, self).__init__()

    @abc.abstractmethod
    def assay(self, *args, **kwargs):
        raise NotImplementedError

class DatasetAssayer(Assayer):
    """Simulated assayer based on dataset.

    Parameters
    ----------
    dataset : Dataset

    Methods
    -------
    assay(dataset)
        Assay a dataset (persumably without `y`).

    Examples
    --------
    >>> import malt
    >>> dataset = malt.data.collections.linear_alkanes(5)
    >>> dataset_assayer = malt.agents.assayer.DatasetAssayer(dataset)
    >>> assayed_dataset = dataset_assayer.assay(dataset)
    >>> assert assayed_dataset == dataset

    """

    def __init__(self, dataset: Dataset):
        super(DatasetAssayer, self).__init__()
        self.dataset = dataset

    def assay(self, dataset):
        """ Assay based on a given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to assay.

        Returns
        -------
        dataset : Dataset
            Assayed dataset, with all `y` annotated.

        """
        for point in dataset:
            assert point in self.dataset
            point.y = self.dataset[point].y
            point.extra = self.dataset[point].extra

        return dataset
