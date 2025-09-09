import torch
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, dataloader, optimizer, device, epochs, experiment_name="default_exp"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.experiment_name = experiment_name

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

    def save_checkpoint(self, epoch_num):
        save_path = f"checkpoints/{self.experiment_name}/model_epoch_{epoch_num}.pt"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the state dict
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

    def train(self):
        print("Starting training...")
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            
            # Save a checkpoint every 10 epochs (can use validation loss as a metric later)
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
        
        # Always save the final model
        self.save_checkpoint("final")
        print("Training complete.")