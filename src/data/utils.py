from typing import Optional
import torch
from torch.utils.data import DataLoader

def make_generator_if_seed(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def make_dataloaders(train_ds, test_ds, cfg):
    pin_memory = bool(cfg.pin_memory and torch.cuda.is_available())
    persistent = bool(cfg.persistent_workers and cfg.num_workers > 0)
    gen = make_generator_if_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        generator=gen
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent
    )
    return train_loader, test_loader