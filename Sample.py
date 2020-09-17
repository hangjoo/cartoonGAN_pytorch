import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Parameters ==================================================================
animation_name = "Weathering_with_You"

train_PATH = "./Data/train"
pre_PATH = "./Saved_model/pretrained_generator.pth"
gen_PATH = "./Saved_model/generator_weight.pth"
dis_PATH = "./discriminator_weight.pth"

pretrain_save_PATH = "./Result/Pretraining"
os.makedirs(pretrain_save_PATH, exist_ok=True)

train_save_PATH = "./Result/Training"
os.makedirs(train_save_PATH, exist_ok=True)

pretrained = True

batch_size = 8
train_epoch = 100
pre_train_epoch = 10

lrG = 0.0002
lrD = 0.0002

optim_beta1 = 0.5
optim_beta2 = 0.999

con_lambda = 10


# Load Data Sets ==============================================================
real_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

cartoon_transform = transforms.Compose([
    transforms.Resize(size=(int(256 * 1.5), int(256 * 1.5))),
    transforms.RandomCrop(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_real = ImageFolder("./Data/train/real", real_transform)
data_cartoon = ImageFolder("./Data/train/cartoon", cartoon_transform)
data_no_edge = ImageFolder("./Data/train/cartoon_none_edge", cartoon_transform)

loader_real = DataLoader(data_real, batch_size=batch_size,
                         shuffle=True, drop_last=True)
loader_cartoon = DataLoader(
    data_cartoon, batch_size=batch_size, shuffle=True, drop_last=True)
loader_no_edge = DataLoader(
    data_no_edge, batch_size=batch_size, shuffle=True, drop_last=True)


print("loader_real =====")
print(len(loader_real))
for idx, data in enumerate(loader_real):
    print(data[0].size())
    print(data[1])
    if idx == 0:
        break

print("loader_cartoon =====")
print(len(loader_cartoon))
for idx, data in enumerate(loader_cartoon):
    print(data[0].size())
    print(data[1])
    if idx == 0:
        break

print("loader_no_edge =====")
print(len(loader_no_edge))
for idx, data in enumerate(loader_no_edge):
    print(data[0].size())
    print(data[1])
    if idx == 0:
        break
