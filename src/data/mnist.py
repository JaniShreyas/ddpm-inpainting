from torchvision import datasets, transforms
from .config import DataConfig
from .utils import make_dataloaders


def get_stats():
    return (0.1307,), (0.3081,)

MNIST_MEAN, MNIST_STD = get_stats()


def default_transform(cfg: DataConfig | None = None):
    # fallback config so callers can pass None
    if cfg is None:
        cfg = DataConfig()

    # if user explicitly provided a transform, respect it
    if cfg.transform is not None:
        return cfg.transform

    t_list = []

    # if an image_size is set, add a resize (must be before ToTensor)
    if cfg.image_size:
        size = cfg.image_size
        # torchvision accepts int or (H, W)
        # if you want to preserve aspect ratio and then crop to exact size,
        # you could do: Resize(max(size)) + CenterCrop(size)
        if isinstance(size, int):
            t_list.append(transforms.Resize(size))
        else:
            t_list.append(transforms.Resize(size))  # (H, W) -> exact size

    # core transforms
    t_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    return transforms.Compose(t_list)


def get_datasets(cfg: DataConfig) -> tuple:
    transform = cfg.transform or default_transform(cfg)
    train_ds = datasets.MNIST(
        root=cfg.data_root, train=True, transform=transform, download=cfg.download
    )
    test_ds = datasets.MNIST(
        root=cfg.data_root, train=False, transform=transform, download=cfg.download
    )
    return train_ds, test_ds


def get_dataloaders(cfg: DataConfig):
    train_ds, test_ds = get_datasets(cfg)
    return make_dataloaders(train_ds, test_ds, cfg)
