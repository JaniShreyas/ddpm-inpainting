from src.models.backbones.unet import UNet
from src.models.backbones.unet_attention import UNetWithAttention
from src.models.backbones.jit import JiT
from src.models.backbones.edm_precond import EDMPrecond

from src.models.ddpm import DiffusionModel
from src.models.autoencoder_kl import AutoEncoderKL
from src.models.edm import EDMModel

import torch.nn as nn

# Builder functions

def create_ddpm_unet_base(config):
    backbone = UNet(**config.model.backbone)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_ddpm_unet_attention(config):
    backbone = UNetWithAttention(**config.model.backbone, image_size=config.dataset.image_size)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_autoencoder_kl(config):
    model = AutoEncoderKL(config=config)
    return model

def create_ddpm_jit(config):
    backbone = JiT(**config.model.backbone)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_edm_song_unet(config):
    # The EDMPrecond wrapper will handle the preconditioning and just call the SongUNet inside it
    backbone = EDMPrecond(**config.model.backbone)
    model = EDMModel(backbone=backbone, config=config)
    return model

# Model Registry
MODEL_REGISTRY = {
    "ddpm_unet_base": create_ddpm_unet_base,
    "ddpm_unet_attention": create_ddpm_unet_attention,
    "autoencoder_kl": create_autoencoder_kl,
    "ddpm_jit": create_ddpm_jit,
    "edm_song_unet": create_edm_song_unet
}

def get_model(config) -> nn.Module:
    name = config.model.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    builder_fn = MODEL_REGISTRY[name]
    return builder_fn(config)