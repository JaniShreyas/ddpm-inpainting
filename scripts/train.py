import os
import mlflow.artifacts
import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

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

    start_epoch = 1
    checkpoint = None
    
    # Resume logic
    if config.resume_from_run_id:
        run_id = config.resume_from_run_id
        print(f"Resuming training from MLFlow Run ID: {run_id}")

        # Getting new total epochs to run
        new_total_epochs = config.training.epochs

        # Download the config from the old run
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config/config.yaml")
        with open(local_path, 'r') as file:
            # Overwrite current config
            config = OmegaConf.create(yaml.safe_load(file))

        # Setting the new total epochs to run
        config.training.epochs = new_total_epochs
        
        # Download the latest checkpoint
        checkpoint_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="checkpoints/latest_checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        start_epoch = checkpoint["epoch"] + 1

        # Start logging to the existing MLFlow run
        mlflow.start_run(run_id=run_id)
        print(f"Resuming in existing MLFlow run: {run_id}")

    else:
        # Start new run
        mlflow.set_experiment(config.experiment_name)
        run = mlflow.start_run()
        print(f"Started new MLFlow run: {run.info.run_id}")

        # Log config for new run
        config_path = "config.yaml"
        OmegaConf.save(config, config_path)
        mlflow.log_artifact(config_path, artifact_path="config")
        os.remove(config_path)
        mlflow.log_params(flatten_config(OmegaConf.to_container(config, resolve=True, throw_on_missing=True)))


    set_seed(config.seed)

    # Setup dataloaders
    data_config = DataConfig(**config.dataset)
    train_dataloader, test_dataloader = get_dataloaders(data_config)

    # Setup model
    model = DiffusionModel(model_config=config.model).to(config.device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6)

    ema_state_dict = None

    # Load state from checkpoint if resuming
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ema_state_dict = checkpoint["ema_model_state_dict"]


    # Instantiate and run the trainer
    trainer = Trainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        get_stats=get_stats,
        config=config,
        start_epoch=start_epoch,
        ema_state_dict=ema_state_dict,
    )

    trainer.train()
    mlflow.end_run()


if __name__ == "__main__":
    main()
