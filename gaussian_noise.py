import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class addNoise(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, sigma):
        super(addNoise, self).__init__()
        self.sigma = sigma


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover
        noised_and_cover = noised_image + (self.sigma ** 2) * torch.randn_like(noised_image)

        return noised_and_cover
