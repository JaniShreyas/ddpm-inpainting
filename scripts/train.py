import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

# Replace with factory get_dataloaders to fit with config yaml
from src.utils.set_seed import set_seed
from src.data.mnist import get_dataloaders, get_stats

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

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    set_seed(config.seed)

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        print(f"Started MLFlow Run ID: {run.info.run_id}")

        mlflow.log_params(flatten_config(OmegaConf.to_container(config, resolve=True, throw_on_missing=True)))

        config_name = "config.yaml"
        OmegaConf.save(config, config_name)
        mlflow.log_artifact(config_name, "config")

        # Setup dataloaders
        data_config = DataConfig(**config.dataset)
        train_dataloader, test_dataloader = get_dataloaders(data_config)

        # Setup model
        model = DiffusionModel(model_config=config.model)

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)

        # Instantiate and run the trainer
        trainer = Trainer(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            get_stats=get_stats,
            config=config
        )

        trainer.train()


if __name__ == "__main__":
    main()
