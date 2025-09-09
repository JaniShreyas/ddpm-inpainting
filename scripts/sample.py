import torch
from src.models.ddpm import DiffusionModel
from torchvision.utils import save_image


def main():
    # Read config [FIX]
    config = {
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 32,
            "channel_multipliers": (1, 2, 4),
        }
    }

    # Instantiate the model
    model = DiffusionModel(model_config=config['model'])

    # Load a trained checkpoint
    checkpoint_path = "path/temporary/trained_model.pt"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Call the sample method
    generated_images = model.sample(num_images=8, image_size=32)

    # Save the images
    # Temporarily overwrites old saves / saves to same location. Need to change later [FIX]
    output_path = "output/samples/generated_grid.png"
    save_image(generated_images, output_path, nrow=4)
    print(f"Saved generated images to {output_path}")

if __name__ == "__main__":
    main()