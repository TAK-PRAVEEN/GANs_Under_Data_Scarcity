def discriminator_loss(real_scores, fake_scores):
    return -(real_scores.mean() - fake_scores.mean())

def generator_loss(fake_scores):
    return -fake_scores.mean()
