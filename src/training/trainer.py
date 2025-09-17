import torch
import mlflow
import torch.utils.data.dataloader
from tqdm import tqdm
import os
from torchvision.utils import save_image
from copy import deepcopy


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
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
        self.dataloader = dataloader
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

    def _update_ema_weights(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param, alpha=(1-self.ema_decay))

    def _train_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0.0

        # tqdm for progress bar
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch_num}")

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
            loss = self.model(clean_images)

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Update ema weights after every step
            self._update_ema_weights()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch_num} - Average loss: {avg_loss:.4f}")

        # Log metrics for MLFlow
        mlflow.log_metric("avg_loss", avg_loss, step=epoch_num)

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

    def sample_and_log_images(self, epoch_num):
        self.model.eval()
        with torch.no_grad():
            generated_images = self.ema_model.sample(
                num_images=self.config["sampling"]["num_images"],
                image_size=self.config["dataset"]["image_size"],
                get_stats=self.get_stats,
                device=self.device,
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
                self.sample_and_log_images(epoch)

        # Always save the final model
        self._save_and_log_checkpoint(self.epochs, is_final=True)
        self.sample_and_log_images("final")
        print("Training complete.")
