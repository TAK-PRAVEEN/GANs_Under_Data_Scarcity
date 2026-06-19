from models.generator import Generator
from models.discriminator import Discriminator

def build_dcgan(z_dim=100, img_channels=3):
    G = Generator(z_dim=z_dim, img_channels=img_channels)
    D = Discriminator(img_channels=img_channels)
    return G, D
