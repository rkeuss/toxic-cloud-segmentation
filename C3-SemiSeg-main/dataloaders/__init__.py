from .ijmond_seg import IJmondSegDataset
from .unlabeled_data import IJmondUnlabeledDataset, RiseDataset

__factory = {
             'IJmondSegDataset': IJmondSegDataset,
             'IJmondUnlabeledDataset': IJmondUnlabeledDataset,
             'RiseDataset': RiseDataset,
            }

def names():
    return sorted(__factory.keys())

def create_dataloader(name, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'pitts', 'tokyo'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown loader:", name)
    return __factory[name]( *args, **kwargs)
