from Model import VGG19 as my_vgg19
from torchvision.models.vgg import vgg19
import torch

tgt_vgg19 = my_vgg19()
src_vgg19 = vgg19(pretrained=True)

print("My VGG19 Model :")
print(tgt_vgg19)

print("\n" + "=" * 90 + "\n")

print("Pretrained VGG19 Model : ")
print(src_vgg19)

tgt_vgg19_dict = tgt_vgg19.state_dict()
src_vgg19_dict = src_vgg19.state_dict()
src_vgg19_dict = {key: val for key, val in src_vgg19_dict.items() if key in tgt_vgg19_dict}
tgt_vgg19_dict.update(src_vgg19_dict)
tgt_vgg19.load_state_dict(tgt_vgg19_dict)

print("\n", "=" * 90, "\n")

print("My VGG19's state_dict :")
for param_tensor in tgt_vgg19_dict:
    print(param_tensor, "\t", tgt_vgg19_dict[param_tensor].size())

print("\n" + "=" * 90 + "\n")

print("Pretrained VGG19's state_dict :")
for param_tensor in src_vgg19_dict:
    print(param_tensor, "\t", src_vgg19_dict[param_tensor].size())

print("\n" + "=" * 90 + "\n")

torch.save(tgt_vgg19.state_dict(), "./pretrained_vgg19.pt")
print("VGG19 Weight Saved.")
