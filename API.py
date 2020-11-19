import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
import torchvision.transforms.functional as F

from Models import Generator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

Generator_PATH = "./Saved_model/generator_weight.pth"

G = Generator(init_weights_path=Generator_PATH)
G.to(device)
G.eval()
print("Cartoon Image Generator Model loaded.")


def preprocessing(img):
    tensor_img = F.to_tensor(img)  # trans to torch tensor type.
    normalized_img = F.normalize(tensor_img, 0.5, 0.5, False)  # normalize [-1, 1] ranges and make tensors be (original value - mean)/ std
    tensor_img = normalized_img.view([-1, normalized_img.shape[0], normalized_img.shape[1], normalized_img.shape[2]])  # make tensor's dim 3 to 4.
    return tensor_img


def cartoon_translate(img, preprocess=True):
    """
    input:
        imgs: numpy image array. it has (height, width, channels) shapes. its dtype is int32 and has [0, 255] ranges. and channels must be RGB orders(not BGR).
        preprocessing: parameter for preprocessing to use our cartoon gan model. it works normalizing and transposes to torch tensor.

    return:
        ndarray dtype returns. its shapes are (height, width, channels) and each shape dtype is float32 and [0, 1] ranges.
    """
    with torch.no_grad():
        pre_img = preprocessing(img).to(device)
        trans_img = G(pre_img)
        return (trans_img[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2


if __name__ == "__main__":
    pass
