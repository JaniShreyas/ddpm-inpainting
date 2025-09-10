import torch
import mlflow
from tqdm import tqdm
import os
from torchvision.utils import save_image

class Trainer:
    def __init__(self, model, dataloader, optimizer, config):

        self.config = config
        self.device = config["device"]
        self.epochs = config["training"]["epochs"]
        self.experiment_name = config["experiment_name"]
        self.sample_every_n_epochs = config["sampling"]["sample_every_n_epochs"]

        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optimizer

    def _train_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0.0

        # tqdm for progress bar
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch_num}")

        for batch in progress_bar:
            # The data might have (image, label) or just (image,) so deal with both
            clean_images = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            
            # Optimizer zero grad
            self.optimizer.zero_grad()

            # Model directly returns loss (helps in generalizing training loop in this ddpm case)
            loss = self.model(clean_images)

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss = loss.item())
        
        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch_num} - Average loss: {avg_loss:.4f}")
        
        # Log metrics for MLFlow
        mlflow.log_metric("avg_loss", avg_loss, step=epoch_num)


    def _save_and_log_checkpoint(self, epoch_num, is_final=False):
        temp_checkpoint_path = f"temp_checkpoint_epoch_{epoch_num}.pt"
        torch.save(self.model.state_dict(), temp_checkpoint_path)

        # Log as artifact
        mlflow.log_artifact(temp_checkpoint_path, artifact_path="checkpoints")

        # If it is final, save as mlflow model as well
        if is_final:
            mlflow.pytorch.log_model(self.model, name="model")
            print("Final model saved in MLFlow model format")

        # Clean up temporary file
        os.remove(temp_checkpoint_path)
        print(f"Logged checkpoints for epoch {epoch_num} to MLFlow")


    def sample_and_log_images(self, epoch_num):
        self.model.eval()
        with torch.no_grad():
            generated_images = self.model.sample(
                num_images=self.config["sampling"]["num_images"],
                image_size=self.config["dataset"]["image_size"],
                device=self.device
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
        print("Starting training...")
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            
            # Save a checkpoint every 10 epochs (can use validation loss as a metric later)
            if epoch % self.sample_every_n_epochs == 0:
                self._save_and_log_checkpoint(epoch)
                self.sample_and_log_images(epoch)
        
        # Always save the final model
        self._save_and_log_checkpoint("final", is_final=True)
        self.sample_and_log_images("final")
        print("Training complete.")