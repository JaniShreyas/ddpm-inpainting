import torch
import mlflow

# Replace with factory get_dataloaders to fit with config yaml
from src.data.mnist import get_dataloaders

from src.models.ddpm import DiffusionModel
from src.training.trainer import Trainer
from src.data.config import DataConfig


def flatten_config(config):
    """Flattens a nested dictionary for MLflow logging."""
    flat_params = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_params[f"{key}.{sub_key}"] = sub_value
        else:
            flat_params[key] = value
    return flat_params

def main():
    # Read config
    # Temporary config setup here. Replace with Hydra config management [FIX]
    config = {
        "experiment_name": "ddpm_mnist_base",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # "seed": 42, # Not yet implemented to use this. Random for now
        "dataset": {"name": "mnist", "batch_size": 128, "image_size": 32},
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 32,
            "channel_multipliers": (1, 2),
            "time_emb_dim": 128
        },
        "training": {
            "epochs": 1,
            "lr": 1e-4
        },
        "sampling": {
            "num_images": 16,
            "sample_every_n_epochs": 5
        }
    }


    mlflow.set_experiment(config['experiment_name'])

    with mlflow.start_run():
        mlflow.log_params(flatten_config(config))

        # Setup dataloaders
        data_config = DataConfig(**config["dataset"])
        train_dataloader, test_dataloader = get_dataloaders(data_config)

        # Setup model
        model = DiffusionModel(model_config=config["model"])

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

        # Instantiate and run the trainer
        trainer = Trainer(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            config=config
        )

        trainer.train()


if __name__ == "__main__":
    main()
