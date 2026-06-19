from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path

def get_dataloader(split_path: str, batch_size=32, num_workers=2):
    split_path = Path(split_path)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # for Tanh output
    ])

    train_data = ImageFolder(root=str(split_path / "train"), transform=transform)

    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return loader
