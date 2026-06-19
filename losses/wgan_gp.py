import torch

def gradient_penalty(D, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = D(interpolated)
    grad = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    grad = grad.view(batch_size, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp
