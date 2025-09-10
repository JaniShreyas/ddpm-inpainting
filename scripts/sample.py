import argparse
import mlflow.artifacts
import mlflow.pytorch
import torch
import mlflow
import yaml
from src.models.ddpm import DiffusionModel
from torchvision.utils import save_image
import os


def main(run_id: str):
    print(f"Loading model from MLFlow Run ID: {run_id}")

    config_name = "config/config.yaml"
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=config_name, dst_path="."
    )
    model_uri = f"runs:/{run_id}/model"

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    os.remove("config.yaml")

    device = config["device"]

    model_loaded = mlflow.pytorch.load_model(model_uri, map_location=device)

    model = DiffusionModel(model_config=config["model"]).to(device)
    model.load_state_dict(model_loaded.state_dict())
    model.eval()
    print(f"Model loaded successfully")

    # Call the sample method
    generated_images = model.sample(
        num_images=config["sampling"]["num_images"],
        image_size=config["dataset"]["image_size"],
        device=device,
    )

    # Save the images
    # Temporarily overwrites old saves / saves to same location. Need to change later [FIX]
    output_path = "output/samples_from_runs/"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_path = f"{output_path}/{run_id}_grid.png"
    save_image(generated_images, save_path, nrow=4)
    print(f"Saved generated images to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from a trained DDPM using an MLflow Run ID."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="The MLflow Run ID of the model to sample from.",
    )
    args = parser.parse_args()

    main(args.run_id)
