import torch

def discriminator_loss(real_logits, fake_logits):
    return 0.5 * (torch.mean((real_logits - 1)**2) +
                  torch.mean(fake_logits**2))

def generator_loss(fake_logits):
    return 0.5 * torch.mean((fake_logits - 1)**2)
