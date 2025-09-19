# Factory to get desired dataloader

from .mnist import get_dataloaders as get_mnist_dataloaders, get_stats as get_mnist_stats
from .cifar10 import get_dataloaders as get_cifar10_dataloaders, get_stats as get_cifar10_stats
from .config import DataConfig
from torch.utils.data import DataLoader 

# The dataset registry
DATASET_REGISTRY = {
    "mnist": (get_mnist_dataloaders, get_mnist_stats),
    "cifar10": (get_cifar10_dataloaders, get_cifar10_stats),
}

def get_dataloaders(cfg: DataConfig) -> DataLoader:
    print(f"Loading dataset: {cfg.name}")

    if cfg.name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {cfg.name} is not supported. Available datasets are: {list(DATASET_REGISTRY.keys())}")
    
    dataloader_fn = DATASET_REGISTRY[cfg.name][0]

    return dataloader_fn(cfg)

def get_stats(cfg: DataConfig) -> tuple:
    return DATASET_REGISTRY[cfg.name][1]