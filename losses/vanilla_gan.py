import torch.nn as nn
import torch

bce = nn.BCEWithLogitsLoss()

def discriminator_loss(real_logits, fake_logits):
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)

    real_loss = bce(real_logits, real_labels)
    fake_loss = bce(fake_logits, fake_labels)
    return real_loss + fake_loss

def generator_loss(fake_logits):
    labels = torch.ones_like(fake_logits)
    return bce(fake_logits, labels)
