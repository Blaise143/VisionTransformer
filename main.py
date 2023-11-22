from src.Train import train
from src.HelperFunctions import plot_loss
from src.VisionTransformer import VisionTransformer

if __name__ == "__main__":
    model = VisionTransformer(
        image_size=28,
        patch_size=4,
        embed_dim=8,
        num_layers=8,
        num_heads=4,
        num_classes=10,
        in_channels=1,
    )
    losses, accuracies = train(model)
    plot_loss(losses, accuracies)

