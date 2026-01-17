import torch
import mlflow
import torch.utils.data.dataloader
from tqdm import tqdm
import os
from torchvision.utils import save_image
from copy import deepcopy
import random


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        test_dataset: torch.utils.data.Dataset,
        optimizer: torch.optim.Optimizer,
        get_stats,
        config,
        start_epoch=1,
        ema_state_dict=None,
    ):
        # Config and device setup
        self.config = config
        self.device = config["device"]
        self.epochs = config["training"]["epochs"]
        self.experiment_name = config["experiment_name"]
        self.sample_every_n_epochs = config["sampling"]["sample_every_n_epochs"]

        # Core Components
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.get_stats = get_stats
        self.start_epoch = start_epoch

        # Setup EMA decay
        self.ema_decay = self.config["training"].get("ema_decay")
        self.ema_model = deepcopy(self.model).eval().requires_grad_(False)
        print(f"EMA enabled with decay rate: {self.ema_decay}")

        # Reload EMA decay
        if ema_state_dict:
            self.ema_model.load_state_dict(ema_state_dict)
            print("Resumed EMA model state from checkpoint")

        self.fixed_sample_batch = next(iter(self.train_dataloader))
        # Handle cases where dataloader returns (data, label)
        if isinstance(self.fixed_sample_batch, (list, tuple)):
            self.fixed_sample_batch = self.fixed_sample_batch[0]
        self.fixed_sample_batch = self.fixed_sample_batch.to(self.device)
        # Take only the number of images we want to sample
        num_images = self.config["sampling"]["num_images"]
        self.fixed_sample_batch = self.fixed_sample_batch[:num_images]

    def _update_ema_weights(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param, alpha=(1-self.ema_decay))

    def _train_epoch(self, epoch_num):
        self.model.train()
        losses_sum = {"loss": 0.0}

        # tqdm for progress bar
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_num}")

        for batch in progress_bar:
            # The data might have (image, label) or just (image,) so deal with both
            clean_images = (
                batch[0].to(self.device)
                if isinstance(batch, (list, tuple))
                else batch.to(self.device)
            )

            # Optimizer zero grad
            self.optimizer.zero_grad()

            # Model directly returns loss (helps in generalizing training loop in this ddpm case)
            losses = self.model(clean_images)
            loss = losses["loss"]

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Update ema weights after every step
            self._update_ema_weights()

            progress_bar.set_postfix(loss=loss.item())

            for loss_type, value in losses.items():
                if loss_type in losses_sum:
                    losses_sum[loss_type] += value.item()
                else:
                    losses_sum[loss_type] = value.item()


        avg_losses = {loss_type: (total_value / len(self.train_dataloader)) for loss_type, total_value in losses_sum.items()}
        print(f"Epoch {epoch_num} - Average loss: {avg_losses["loss"]:.4f}")

        # Log metrics for MLFlow
        for loss_type in avg_losses:
            mlflow.log_metric(f"avg_{loss_type}", avg_losses[loss_type], step=epoch_num)

    def _save_and_log_checkpoint(self, epoch_num, is_final=False):
        base_checkpoint_artifact_dir = "checkpoints"
        checkpoint = {
            "epoch": epoch_num,
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        temp_checkpoint_path = f"checkpoint_epoch_{epoch_num}.pt"
        torch.save(checkpoint, temp_checkpoint_path)

        # Log as artifact
        mlflow.log_artifact(temp_checkpoint_path, artifact_path=base_checkpoint_artifact_dir)

        # final_checkpoint variable for easy access
        latest_checkpoint_path = "latest_checkpoint.pt"
        torch.save(checkpoint, latest_checkpoint_path)
        mlflow.log_artifact(latest_checkpoint_path, artifact_path=base_checkpoint_artifact_dir)

        # If it is final, save the ema model as mlflow model as well
        if is_final:
            mlflow.pytorch.log_model(self.ema_model, name="model")
            print("Final EMA model saved in MLFlow model format")

        # Clean up temporary file
        os.remove(temp_checkpoint_path)
        os.remove(latest_checkpoint_path)
        print(f"Logged checkpoints for epoch {epoch_num} to MLFlow")

    def _validate_checkpoint(self, epoch_num, if_final=False):
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(self.test_dataloader, desc=f"val_{epoch_num}")
            losses_sum = {"loss": 0.0}
            for batch in progress_bar:
                clean_images = (
                    batch[0].to(self.device)
                    if isinstance(batch, (list, tuple))
                    else batch.to(self.device)
                )

                losses = self.model(clean_images)

                for loss_type, value in losses.items():
                    if loss_type in losses_sum:
                        losses_sum[loss_type] += value.item()
                    else:
                        losses_sum[loss_type] = value.item()

            avg_losses = {loss_type: (total_value / len(self.train_dataloader)) for loss_type, total_value in losses_sum.items()}
            print(f"Average validation loss: {avg_losses["loss"]}")

            for loss_type in avg_losses:
                mlflow.log_metric(f"val_avg_{loss_type}", avg_losses[loss_type], step=epoch_num)
            


    def sample_and_log_images(self, epoch_num):
        self.model.eval()
        with torch.no_grad():
            generated_images = self.ema_model.sample(
                num_images=self.config["sampling"]["num_images"],
                image_size=self.config["dataset"]["image_size"],
                get_stats=self.get_stats,
                device=self.device,
                sample_x=self.fixed_sample_batch,
            )

        # Save temporary file
        temp_path = f"temp_sample_epoch_{epoch_num}.png"
        save_image(generated_images, temp_path, nrow=4)

        # Log as an artifact to mlflow
        mlflow.log_artifact(temp_path, artifact_path="samples")

        # Clean up temporary file
        os.remove(temp_path)
        print(f"Logged sample images for epoch {epoch_num} to MLFlow")

    def train(self):
        print(f"Starting training from epoch {self.start_epoch}...")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

            # Save a checkpoint every 10 epochs (can use validation loss as a metric later)
            if epoch % self.sample_every_n_epochs == 0:
                self._save_and_log_checkpoint(epoch)
                self._validate_checkpoint(epoch)
                self.sample_and_log_images(epoch)

        # Always save the final model
        self._save_and_log_checkpoint(self.epochs, is_final=True)
        self._validate_checkpoint(self.epochs, if_final=True)
        self.sample_and_log_images("final")
        print("Training complete.")
