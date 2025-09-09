import torch
# Replace with factory get_dataloaders to fit with config yaml
from src.data.mnist import get_dataloaders

from src.models.ddpm import DiffusionModel
from src.training.trainer import Trainer
from src.data.config import DataConfig

def main():
    # Read config
    # Temporary config setup here. Replace with Hydra config management
    config = {
        'dataset': {'name': 'mnist', 'batch_size': 128, 'image_size': 32},
        'model': {'in_channels': 1, 'out_channels': 1, 'base_channels': 32, 'channel_multipliers': (1,2)},
        'optimizer': {'lr': 1e-4},
        'training': {'epochs': 10, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    }

    # Setup dataloaders
    data_config = DataConfig(**config['dataset'])
    train_dataloader, test_dataloader = get_dataloaders(data_config)

    # Setup model
    model = DiffusionModel(model_config=config['model'])
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'])

    # Instantiate and run the trainer
    trainer = Trainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        **config['training']
    )

    trainer.train()

if __name__ == "__main__":
    main()