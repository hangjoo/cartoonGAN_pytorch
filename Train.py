import time
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from Utils import edge_smoothing
from Model import Generator, Discriminator, VGG19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

# Parameters ==================================================================
animation_name = "Naruto_3"

data_PATH = "./Train/Data"
vgg_PATH = "./Saved_model/pretrained_vgg19.pth"
pre_PATH = "./Saved_model/pretrained_generator.pth"
gen_PATH = "./Saved_model/generator_weight.pth"
dis_PATH = "./Saved_model/discriminator_weight.pth"

smoothed = False
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
real_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

cartoon_transform = transforms.Compose(
    [
        transforms.Resize(size=(int(256 * 1.5), int(256 * 1.5))),
        transforms.RandomCrop(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

if not smoothed:
    print("======= Edge Smoothing Start =======")
    edge_smoothing(
        src_path=os.path.join(data_PATH, "cartoon/1"),
        dst_path=os.path.join(data_PATH, "edge_smoothing/0"),
    )

data_real = ImageFolder(os.path.join(data_PATH, "real"), real_transform)
data_cartoon = ImageFolder(os.path.join(data_PATH, "cartoon"), cartoon_transform)
data_no_edge = ImageFolder(os.path.join(data_PATH, "edge_smoothing"), cartoon_transform)
data_validation = ImageFolder(os.path.join(data_PATH, "validation"), real_transform)

loader_real = DataLoader(data_real, batch_size=batch_size, shuffle=True, drop_last=True)
loader_cartoon = DataLoader(
    data_cartoon, batch_size=batch_size, shuffle=True, drop_last=True
)
loader_no_edge = DataLoader(
    data_no_edge, batch_size=batch_size, shuffle=True, drop_last=True
)
loader_validation = DataLoader(
    data_validation, batch_size=1, shuffle=False, drop_last=False
)

# Models ======================================================================
G = Generator()
D = Discriminator()
P_VGG19 = VGG19(weight_PATH=vgg_PATH)

G.to(device)
D.to(device)
P_VGG19.to(device)

G.train()
D.train()
P_VGG19.eval()

# Loss ========================================================================
BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)

# Optimizer & Scheduler =======================================================
G_optimizer = optim.Adam(G.parameters(), lr=lrG, betas=(optim_beta1, optim_beta2))
D_optimizer = optim.Adam(D.parameters(), lr=lrD, betas=(optim_beta1, optim_beta2))

G_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer=G_optimizer,
    milestones=[train_epoch // 2, train_epoch // 4 * 3],
    gamma=0.1,
)
D_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer=D_optimizer,
    milestones=[train_epoch // 2, train_epoch // 4 * 3],
    gamma=0.1,
)

# Pre-train Generator =========================================================
if not pretrained:
    print("======= Generator Pretraining Start =======")
    start_time = time.time()
    pretrain_save_PATH = "./Train/Pretraining"
    os.makedirs(pretrain_save_PATH, exist_ok=True)
    for epoch_idx in range(1, pre_train_epoch + 1):
        G.train()
        epoch_start_time = time.time()
        Recon_losses = []
        for x, _ in loader_real:
            x = x.to(device)
            trans_x = G(x)

            G_optimizer.zero_grad()
            x_feature = P_VGG19((x + 1) / 2)
            trans_feature = P_VGG19((trans_x + 1) / 2)

            Recon_loss = 10 * L1_loss(trans_feature, x_feature.detach())
            Recon_losses.append(Recon_loss.item())

            Recon_loss.backward()
            G_optimizer.step()

        per_epoch_time = time.time() - epoch_start_time
        print(
            "[%03d/%03d] - time: %.2f | Recon Loss: %.3f"
            % (
                epoch_idx,
                pre_train_epoch,
                per_epoch_time,
                torch.mean(torch.FloatTensor(Recon_losses)),
            )
        )

        # Save Model in each epoch.
        torch.save(
            G.state_dict(),
            os.path.join(pretrain_save_PATH, "Epoch_%03d_Generator.pth" % (epoch_idx)),
        )

        # Save original and generated image in each epoch.
        with torch.no_grad():
            G.eval()
            for idx, (x, _) in enumerate(loader_real):
                x = x.to(device)
                G_recon = G(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                plt.imsave(
                    os.path.join(
                        pretrain_save_PATH,
                        "Epoch_%03d_Reconstruct_%03d.png" % (epoch_idx, idx + 1),
                    ),
                    (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2,
                )
                if idx == 4:
                    break

    total_time = time.time() - start_time

    torch.save(G.state_dict(), pre_PATH)

else:
    print("======= Generator already pretrained =======")
    G.load_state_dict(torch.load(pre_PATH))

# Train =======================================================================
print("======= Training start =======")

start_time = time.time()

train_save_PATH = os.path.join("./Train/Training", animation_name)
os.makedirs(train_save_PATH, exist_ok=True)

progress_hist = open(os.path.join(train_save_PATH, animation_name + "_progress.txt", "a"))

checkpoint = 0

real = torch.ones(batch_size, 1, 256 // 4, 256 // 4).to(device)
fake = torch.zeros(batch_size, 1, 256 // 4, 256 // 4).to(device)
edge = torch.full(size=(batch_size, 1, 256 // 4, 256 // 4), fill_value=0.2).to(device)

for idx in range(1, train_epoch + 1):
    if os.path.isfile(
        os.path.join(
            train_save_PATH, "Epoch_%03d" % (idx), "Epoch_%03d_Generator.pth" % (idx)
        )
    ) and os.path.isfile(
        os.path.join(
            train_save_PATH,
            "Epoch_%03d" % (idx),
            "Epoch_%03d_Discriminator.pth" % (idx),
        )
    ):
        Gen_checklist = torch.load(
            os.path.join(
                train_save_PATH,
                "Epoch_%03d" % (idx),
                "Epoch_%03d_Generator.pth" % (idx),
            )
        )
        Dis_checklist = torch.load(
            os.path.join(
                train_save_PATH,
                "Epoch_%03d" % (idx),
                "Epoch_%03d_Discriminator.pth" % (idx),
            )
        )

        G.load_state_dict(Gen_checklist["model_state_dict"])
        G_optimizer.load_state_dict(Gen_checklist["optimizer_state_dict"])

        D.load_state_dict(Dis_checklist["model_state_dict"])
        D_optimizer.load_state_dict(Dis_checklist["optimizer_state_dict"])

        checkpoint = Gen_checklist["epoch"]

        print(
            "%03d-Epoch model save file exists. Epoch_%03d_Generator.pth, Epoch_%03d_Discriminator.pth is loaded."
            % (idx, idx, idx)
        )


for epoch_idx in range(1, train_epoch + 1):
    epoch_start_time = time.time()

    if epoch_idx <= checkpoint:
        print("[%d/%d] Already trained in past." % (epoch_idx, train_epoch))
        continue

    G.train()

    Dis_losses = []
    Adv_losses = []
    Con_losses = []
    Gen_losses = []

    for (x, _), (y, _), (e, _) in zip(loader_real, loader_cartoon, loader_no_edge):
        x, y, e = x.to(device), y.to(device), e.to(device)

        # Train Discriminator
        if epoch_idx % 3 == 1:
            D_optimizer.zero_grad()

            D_real = D(y)
            D_real_loss = BCE_loss(D_real, real)

            trans_x = G(x)
            D_fake = D(trans_x)
            D_fake_loss = BCE_loss(D_fake, fake)

            D_none_edge = D(e)
            D_none_edge_loss = BCE_loss(D_none_edge, fake)

            Dis_loss = D_real_loss + D_fake_loss + D_none_edge_loss
            Dis_losses.append(Dis_loss.item())

            Dis_loss.backward()
            D_optimizer.step()

        # Train Generator
        G_optimizer.zero_grad()

        trans_x = G(x)
        D_fake = D(trans_x)
        Adv_loss = BCE_loss(D_fake, real)

        x_feature = P_VGG19((x + 1) / 2)
        trans_feature = P_VGG19((trans_x + 1) / 2)
        Con_loss = con_lambda * L1_loss(trans_feature, x_feature.detach())

        Gen_loss = Adv_loss + Con_loss

        Adv_losses.append(Adv_loss.item())
        Con_losses.append(Con_loss.item())
        Gen_losses.append(Gen_loss.item())

        Gen_loss.backward()
        G_optimizer.step()

    G_scheduler.step()
    D_scheduler.step()

    per_epoch_time = time.time() - epoch_start_time
    print(
        "[%d/%d] - time: %.2f | Discriminator Loss: %.3f | Generator Loss: %.3f - Adv Loss : %.3f, Con Loss : %.3f"
        % (
            epoch_idx,
            train_epoch,
            per_epoch_time,
            torch.mean(torch.FloatTensor(Dis_losses)),
            torch.mean(torch.FloatTensor(Gen_losses)),
            torch.mean(torch.FloatTensor(Adv_losses)),
            torch.mean(torch.FloatTensor(Con_losses)),
        )
    )
    progress_hist.write(
        "[%d/%d] - time: %.2f | Discriminator Loss: %.3f | Generator Loss: %.3f - Adv Loss : %.3f, Con Loss : %.3f"
        % (
            epoch_idx,
            train_epoch,
            per_epoch_time,
            torch.mean(torch.FloatTensor(Dis_losses)),
            torch.mean(torch.FloatTensor(Gen_losses)),
            torch.mean(torch.FloatTensor(Adv_losses)),
            torch.mean(torch.FloatTensor(Con_losses)),
        )
    )

    epoch_save_PATH = os.path.join(train_save_PATH, "Epoch_%03d" % epoch_idx)
    os.makedirs(epoch_save_PATH, exist_ok=True)

    with torch.no_grad():
        G.eval()

        for idx, (x, _) in enumerate(loader_validation):
            x = x.to(device)
            trans_x = G(x)

            result = torch.cat((x[0], trans_x[0]), 2)

            plt.imsave(
                os.path.join(
                    epoch_save_PATH,
                    "Epoch_%03d_validation_%03d.png" % (epoch_idx, idx + 1),
                ),
                (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2,
            )

        if epoch_idx % 5 == 0:
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": G.state_dict(),
                    "optimizer_state_dict": G_optimizer.state_dict(),
                },
                os.path.join(epoch_save_PATH, "Epoch_%03d_Generator.pth" % (epoch_idx)),
            )
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": D.state_dict(),
                    "optimizer_state_dict": D_optimizer.state_dict(),
                },
                os.path.join(
                    epoch_save_PATH, "Epoch_%03d_Discriminator.pth" % (epoch_idx)
                ),
            )

total_time = time.time() - start_time

print("======= Congratulations! Training is all done. =======")
print("=======         total time taken : %.2f        =======" % (total_time))

progress_hist.write("======= total time taken : %.2f" % (total_time))
progress_hist.close()
