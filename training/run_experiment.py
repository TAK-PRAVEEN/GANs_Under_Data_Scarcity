import os
import sys
import time
import csv
import torch
from torch.optim import Adam

# Get the directory of the current script (training/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (GANs_Under_Data_Scarcity/)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from models.dcgan import build_dcgan
from src.dataset import get_dataloader
from src.utils import save_samples, count_params

from losses import vanilla_gan, lsgan, wgan, wgan_gp

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(loss_name, split_size, epochs=50, batch_size=32):

    split_path = f"data/splits/{split_size}"
    dataloader = get_dataloader(split_path, batch_size=batch_size)

    G, D = build_dcgan()
    G, D = G.to(device), D.to(device)

    opt_G = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    z_dim = 100

    # WGAN-GP hyperparams
    lambda_gp = 10
    n_critic = 5 if loss_name in ["wgan", "wgan_gp"] else 1

    exp_name = f"dcgan_{loss_name}_{split_size}"
    img_dir = f"experiments/images/{exp_name}"
    log_dir = f"experiments/logs"
    ckpt_dir = f"experiments/checkpoints/{exp_name}"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file = f"{log_dir}/{exp_name}.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "d_loss", "g_loss", "epoch_time", "params_G(M)", "params_D(M)"])

    print(f"\nRunning Experiment: {exp_name}")
    print(f"Data: {split_size}, Loss: {loss_name}, Epochs: {epochs}, Batch: {batch_size}")
    print(f"Params G: {count_params(G):.2f}M, D: {count_params(D):.2f}M\n")

    for epoch in range(1, epochs + 1):
        start = time.time()
        d_loss_total, g_loss_total = 0.0, 0.0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ============================================================
            # 1) Train Discriminator / Critic n_critic times
            # ============================================================
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake_images = G(noise)

                real_out = D(real_images)
                fake_out = D(fake_images.detach())

                if loss_name == "vanilla":
                    d_loss = vanilla_gan.discriminator_loss(real_out, fake_out)

                elif loss_name == "lsgan":
                    d_loss = lsgan.discriminator_loss(real_out, fake_out)

                elif loss_name == "wgan":
                    d_loss = wgan.discriminator_loss(real_out, fake_out)

                    # original WGAN should use weight clipping (optional)
                    for p in D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                elif loss_name == "wgan_gp":
                    gp = wgan_gp.gradient_penalty(D, real_images, fake_images.detach(), device)
                    d_loss = -(real_out.mean() - fake_out.mean()) + lambda_gp * gp

                else:
                    raise ValueError("Unknown loss")

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # ============================================================
            # 2) Train Generator (one step)
            # ============================================================
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = G(noise)
            fake_out = D(fake_images)

            if loss_name == "vanilla":
                g_loss = vanilla_gan.generator_loss(fake_out)

            elif loss_name == "lsgan":
                g_loss = lsgan.generator_loss(fake_out)

            elif loss_name == "wgan":
                g_loss = wgan.generator_loss(fake_out)

            elif loss_name == "wgan_gp":
                g_loss = -fake_out.mean()

            else:
                raise ValueError("Unknown loss")

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()

        end = time.time()
        epoch_time = end - start

        d_avg = d_loss_total / len(dataloader)
        g_avg = g_loss_total / len(dataloader)

        if epoch % 5 == 0:
            save_samples(G, epoch, z_dim, device, out_dir=img_dir, n=64)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, d_avg, g_avg, epoch_time, count_params(G), count_params(D)])

        print(f"Epoch {epoch:03d} | D Loss: {d_avg:.4f} | G Loss: {g_avg:.4f} | Time: {epoch_time:.2f}s")

    torch.save(G.state_dict(), f"{ckpt_dir}/G.pth")
    torch.save(D.state_dict(), f"{ckpt_dir}/D.pth")

    print(f"\nDone. Logs saved: {log_file}")
    print(f"Images saved: {img_dir}")
    print(f"Checkpoints saved: {ckpt_dir}")

if __name__ == "__main__":
    train(loss_name="wgan_gp", split_size=25000, epochs=50, batch_size=32)
