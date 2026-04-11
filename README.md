# Generative MLOps: A Reproducible Diffusion Training Framework

This is a robust, professional-grade MLOps framework (hopefully :D) with structure and modularity to add and customize models, algorithms, datasets, metrics, and much more as needed, while keeping everything structured and with apt Experiment tracking and Config management. The primary goal is to engineer a scalable and reproducible pipeline for systematic deep learning research.

The framework is designed to be highly modular and extensible, allowing for easy adaptation to different models (e.g., text-to-image, inpainting) and datasets. It leverages a modern MLOps toolchain, including Hydra for composable configuration management and MLflow for comprehensive experiment tracking, to ensure every result is auditable and reproducible.

The project, as of now, supports MNIST digits and CIFAR10 as datasets, a traditional DDPM training algorithm with DDIM sampling, and backbones for UNet and JiT (from the Back to Basics paper: (https://arxiv.org/pdf/2511.13720). There will be more models and backbones (EDM training, DiT backbone) and datasets (CelebA) added in the future.
The priority will be on models and techniques suitable for training on low-resource environments (like consumer GPUs), and I'll be adding new stuff as I go through research papers.
There will also be some experiment logs, which I'll update here later as of their location in the future.

## Key Features

### Machine Learning Core
- **Algorithmic Flexibility:** From-scratch traditional DDPM training implementation with currently **DDIM sampling** as default. The framework is modularly built to easily accommodate future training methodologies like EDM.
- **Interchangeable Backbones:** Currently supports a highly configurable **U-Net** (with customizable depth/width via YAML, Time-Conditioned Residual Blocks, and Sinusoidal Position Embeddings) and the newly integrated **JiT** architecture (from the *Back to Basics* paper). The structure is designed to seamlessly integrate other important architectures like DiT.
- **Plug-and-Play Datasets:** Out-of-the-box support and standardized pipelines for **MNIST digits** and **CIFAR10**, with infrastructure ready for future high-resolution additions like CelebA or any user-defined custom datasets.
- **Low-Resource Optimization:** A strict engineering priority on implementing models and training techniques that are highly efficient and suitable for consumer-grade GPUs and low-resource environments.
- **Advanced Training Techniques:** Includes industry best practices like Exponential Moving Average (EMA) weights for improved sample quality and stability.

FID is currently used as a metric but others can be easily added in `training/trainer.py` in the appropriate location and called in the main loop `train()`

### MLOps & Engineering Framework
- **Composable Configuration with Hydra:** All experimental parameters are managed in a modular `configs/` directory. This enables rapid, code-free experimentation through simple command-line overrides.
- **Comprehensive Experiment Tracking with MLflow:** Every run is automatically logged. MLflow tracks all hyperparameters, monitors live metrics (loss curves), and versions all output artifacts (models, checkpoints, and sample images).
- **Guaranteed Reproducibility:** The framework uses global seeding and logs the exact configuration for every run, ensuring that any experiment can be perfectly reproduced using its MLflow Run ID.

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

### Sampling from a Trained Model (Currently a little outdated in comparison to the training code and will not work. I'll be fixing it when it is needed :D)

To generate images from a specific experiment, use the `sample.py` script and provide the MLflow Run ID. This guarantees you are using the exact model and configuration from that run.

```
uv run -m scripts.sample --run_id <your_mlflow_run_id>
```




