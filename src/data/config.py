from dataclasses import dataclass
from typing import Optional
import torchvision.transforms as T

@dataclass
class DataConfig:
    name: str = "mnist"             # dataset key used in factory
    data_root: str = "datasets"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    download: bool = True
    shuffle_train: bool = True
    seed: Optional[int] = None
    image_size: Optional[tuple[int, int]] = None
    transform: Optional[T.Compose] = None

    def copy_with(self, **kw):
        """Small helper to produce modified configs."""
        fields = {**self.__dict__, **kw}
        return DataConfig(**fields)