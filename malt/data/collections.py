# =============================================================================
# IMPORTS
# =============================================================================
import dgllife
from malt.data.dataset import Dataset
from malt.point import Point
import copy

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _dataset_from_dgllife(dgllife_dataset):
    idx = 0
    ds = []
    for smiles, g, y in dgllife_dataset:
        point = Point(smiles, g, y.item(), extra={"idx": idx})
        idx += 1
        ds.append(point)

    ds = Dataset(ds)

    return ds


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
COLLECTIONS = [
    "ESOL",
    "FreeSolv",
    "Lipophilicity",
]

for collection in COLLECTIONS:

    def _get_collection():
        from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
        ds = getattr(dgllife.data, collection)(
            smiles_to_bigraph, CanonicalAtomFeaturizer()
        )
        return _dataset_from_dgllife(ds)

    globals()[collection.lower()] = _get_collection

def linear_alkanes(max_carbon=10):
    """A toy dataset with linear alkanes from 1 to `max_carbon` carbons.

    Parameters
    ----------
    max_carbon : int
        Maximum number of carbons in the molecules generated.

    Returns
    -------
    Dataset
        A dataset containing alanes.

    Examples
    --------
    dataset = count_carbons(10)


    """

    return Dataset([Point(idx * "C") for idx in range(1, max_carbon+1)])
