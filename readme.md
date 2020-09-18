# CartoonGAN with pytorch

## Abstract

CartoonGAN implementation code with pytorch framework.

## Environment

- Modules
  - python : 3.6.10
  - pytorch : 1.6.0
  - torchvision : 0.7.0
- GPU
  - k80 (Use Microsoft Azure VM server)
  - Your GPU card must have at least 11GB of VRAM.
- OS
  - Ubuntu 18.04-LTS
- Folder Structure

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
      └── Sample_cartoon
```

## How to train models

1. If you want to train models, save the cartoon image set in "Data/cartoon/1/" directory. and also save the real image set in "Data/real/0" directory.
2. Open the Train.py and edit the parameters at line 20-48.
3. Execute Train.py. (It takes a very long time, so I recommend using it as a background task.)
