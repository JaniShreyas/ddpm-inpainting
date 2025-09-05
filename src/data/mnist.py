from torchvision import datasets, transforms
from .config import DataConfig
from .utils import make_dataloaders

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

def default_transform(cfg: DataConfig = None):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

def get_datasets(cfg: DataConfig) -> tuple:
    transform = cfg.transform or default_transform(cfg)
    train_ds = datasets.MNIST(root=cfg.data_root, train=True, transform=transform, download=cfg.download)
    test_ds = datasets.MNIST(root=cfg.data_root, train=False, transform=transform, download=cfg.download)
    return train_ds, test_ds

def get_dataloaders(cfg: DataConfig):
    train_ds, test_ds = get_datasets(cfg)
    return make_dataloaders(train_ds, test_ds, cfg)