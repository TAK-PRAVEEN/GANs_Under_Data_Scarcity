import os
import torch
from torchvision.utils import save_image

def save_samples(G, epoch, z_dim, device, out_dir="experiments/images", n=64):
    os.makedirs(out_dir, exist_ok=True)
    G.eval()
    with torch.no_grad():
        noise = torch.randn(n, z_dim, 1, 1).to(device)
        fake = G(noise)
        save_image(fake, f"{out_dir}/epoch_{epoch}.png", normalize=True)
    G.train()

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
