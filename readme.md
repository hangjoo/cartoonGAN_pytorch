# CartoonGAN with pytorch

## Abstract

[CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)(in CVPR2018) implementation code with pytorch.

## Environment

### 1.  Modules

    - python : 3.6.10
    - pytorch : 1.6.0
    - torchvision : 0.7.0

### 2.  GPU

    - k80 (Microsoft Azure VM server)
    - Your GPU card must have at least 11GB of VRAM.

### 3.  OS

    - Ubuntu 18.04-LTS

### 4.  Directory Structure

```
  .
  ├── Data  
  │   └── train
  │       ├── cartoon
  │       │   └── 1
  │       ├── edge_smoothing
  │       │   └── 0
  │       └── real
  │           └── 0
  ├── Saved_model
  ├── src
  └── Train
      ├── Pretraining
      └── Training
```

### _"./Data/train/cartoon/1":_

The directory to save cartoon image set to train models.

### _"./Data/train/edge_smoothing/0":_

The directory to save edge smoothed cartoon image set to train models. You don't have to do anything to this directory.

### _"./Data/train/real/0":_

The directory to save real image set to train models.

### _"./Saved_model":_

After the training process is over, the generated weight files are saved in this directory.

### _"./src":_

Only used for other tasks.

### _"./Train/Pretraining":_

The intermediate results and weight files created during the pre-training process are saved.

### _"./Train/Training":_

The intermediate validation results and weight files(every 5 epoch) created during the main-training process are saved.

## How to train models

1. If you want to train models, put the cartoon image set in "Data/cartoon/1/" directory. and put the real image set in "Data/real/0" directory.
2. Open the Train.py and edit the parameters at line 20-41.
3. Execute Train.py. (It takes a very long time.)

## Our train results

We used cartoon image set from *Tom and Jerry* animation to train models. And Flickr 30k dataset was used for real image set. Each label contains about 20,000 images.

### Initialization phase (section [3.3] in paper)

|               Original                |                  Epoch 1                   |                  Epoch 5                   |                  Epoch 10                  |
| :-----------------------------------: | :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
| ![Original_1](./src/001_original.png) | ![Epoch_1_1](./src/001_vgg19_epoch_01.png) | ![Epoch_4_1](./src/001_vgg19_epoch_05.png) | ![Epoch_9_1](./src/001_vgg19_epoch_10.png) |
| ![Original_2](./src/002_original.png) | ![Epoch_1_2](./src/002_vgg19_epoch_01.png) | ![Epoch_4_2](./src/002_vgg19_epoch_05.png) | ![Epoch_9_2](./src/002_vgg19_epoch_10.png) |
| ![Original_3](./src/003_original.png) | ![Epoch_1_3](./src/003_vgg19_epoch_01.png) | ![Epoch_4_3](./src/003_vgg19_epoch_05.png) | ![Epoch_9_3](./src/003_vgg19_epoch_10.png) |
| ![Original_4](./src/004_original.png) | ![Epoch_1_4](./src/004_vgg19_epoch_01.png) | ![Epoch_4_4](./src/004_vgg19_epoch_05.png) | ![Epoch_9_4](./src/004_vgg19_epoch_10.png) |

### After Train

|               Original                |                Converted                 |
| :-----------------------------------: | :--------------------------------------: |
| ![Original_1](./src/001_original.png) | ![converted_1](./src/001_main_train.png) |
| ![Original_2](./src/002_original.png) | ![converted_2](./src/002_main_train.png) |
| ![Original_3](./src/003_original.png) | ![converted_3](./src/003_main_train.png) |
| ![Original_4](./src/004_original.png) | ![converted_4](./src/004_main_train.png) |