import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from Model import Generator, Discriminator, VGG19

from torchvision import transforms
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

Generator_PATH = "./Saved_model/pretrained_generator.pth"

src_PATH = "./Translate/src"
dst_PATH = "./Translate/dst"

G = Generator(weight_PATH=Generator_PATH)
G.to(device)

transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

data_set = ImageFolder(src_PATH, transform)
loader = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False)

with torch.no_grad():
    G.eval()
    
    for idx, (x, _) in enumerate(loader):
        x = x.to(device)
        trans_x = G(x)

        result = torch.cat((x[0], trans_x[0]), 2)

        plt.imsave(
            os.path.join(dst_PATH, "compare_shot_%03d.png" % (idx + 1)),
            (trans_x[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2,
        )
        plt.imsave(
            os.path.join(dst_PATH, "translated_%03d.png" % (idx + 1)),
            (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2,
        )

        print("[%03d/%03d] translated ... " % (idx + 1, len(loader)))

print("All Done !")