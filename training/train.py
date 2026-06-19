import torch
from models.dcgan import build_dcgan
from losses import vanilla_gan
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

G, D = build_dcgan()
G, D = G.to(device), D.to(device)

opt_G = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

z_dim = 100

for epoch in range(epochs):
    for real_images in dataloader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_images = G(noise)

        # Discriminator
        real_logits = D(real_images)
        fake_logits = D(fake_images.detach())

        d_loss = vanilla_gan.discriminator_loss(real_logits, fake_logits)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Generator
        fake_logits = D(fake_images)
        g_loss = vanilla_gan.generator_loss(fake_logits)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
