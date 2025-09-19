# Generative MLOps: A Reproducible Diffusion Training Framework

This project is a from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) built within a robust, professional-grade MLOps framework. The primary goal is not just to build a generative model, but to engineer a scalable and reproducible pipeline for systematic deep learning research.

The framework is designed to be highly modular and extensible, allowing for easy adaptation to different models (e.g., text-to-image, inpainting) and datasets. It leverages a modern MLOps toolchain, including Hydra for composable configuration management and MLflow for comprehensive experiment tracking, to ensure every result is auditable and reproducible.

## Key Features

### Machine Learning Core
- **From-Scratch DDPM Implementation:** The entire diffusion process, including the noise schedule and sampling loop, is built from first principles in PyTorch.
- **Configurable U-Net Architecture:** A flexible U-Net model serves as the denoising backbone. Its depth (channel_multipliers) and width (base_channels) can be configured directly from a YAML file without changing any code.
- **Time-Conditioned Residual Blocks:** The U-Net uses ResidualBlocks for stable training and SinusoidalPositionEmbeddings to effectively condition the model on the current timestep.
- **Advanced Training Techniques:** The pipeline includes best practices like Exponential Moving Average (EMA) weights for improved sample quality.

### MLOps & Engineering Framework
- Composable Configuration with Hydra: All experimental parameters are managed in a modular `configs/` directory. This enables rapid, code-free experimentation through simple command-line overrides.
- Comprehensive Experiment Tracking with MLflow: Every run is automatically logged. MLflow tracks all hyperparameters, monitors live metrics (loss curves), and versions all output artifacts (models, checkpoints, and sample images).
- Guaranteed Reproducibility: The framework uses global seeding and logs the exact configuration for every run, ensuring that any experiment can be perfectly reproduced using its MLflow Run ID.

## Setup instructions

### Install uv
This project uses uv for package management: https://github.com/astral-sh/uv

Follow the instructions in the above repo link or run the following command in powershell to install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then run the following in the repo directory to setup the packages

```powershell
uv sync
```

## Usage

### Training a new model

To start a new training run, simply execute the main training script. Hydra will automatically use the default configuration defined in `configs/config.yaml`.

```
uv run -m scripts.train
```

To run an experiment with different parameters, you can override any value from the command line without editing files.

```
# Example: Train a wider and deeper model with a different learning rate
uv run -m scripts.train model.base_channels=128 training.learning_rate=0.0005
```

### Resuming a training run

To continue a previous run, use the `resume_from_run_id` parameter. Find the Run ID from your MLflow UI.
```
# Example: Continue training run 'abc123def' and train until epoch 200
uv run -m scripts.train resume_from_run_id=abc123def training.epochs=200
```

### Sampling from a Trained Model

To generate images from a specific experiment, use the `sample.py` script and provide the MLflow Run ID. This guarantees you are using the exact model and configuration from that run.

```
uv run -m scripts.sample --run_id <your_mlflow_run_id>
```


