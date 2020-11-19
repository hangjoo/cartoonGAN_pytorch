# CartoonGAN with pytorch

## Abstract

CartoonGAN implementation code with pytorch framework.

## Environment

1. ### Modules

    - python : 3.6.10
    - pytorch : 1.6.0
    - torchvision : 0.7.0

2. ### GPU

    - k80 (Use Microsoft Azure VM server)
    - Your GPU card must have at least 11GB of VRAM.

3. ### OS

    - Ubuntu 18.04-LTS

4. ### Directory Structure

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

The intermediate results and weght files created during the pre-training process are saved.

### _"./Train/Training":_

The intermediate validatino results and weght files(every 5 epoch) created during the main-training process are saved.

## How to train models

1. If you want to train models, put the cartoon image set in "Data/cartoon/1/" directory. and put the real image set in "Data/real/0" directory.
2. Open the Train.py and edit the parameters at line 20-41.
3. Execute Train.py. (It takes a very long time.)

## Our train results

