# CartoonGAN with pytorch

## Abstract

---

CartoonGAN implementation code with pytorch framework.

## Environment

---

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
  .  
  ├── Data  
  │   ├── cartoon_images  
  │   │   └── Sample_cartoon  
  │   ├── real_images  
  │   ├── train  
  │   │   ├── cartoon  
  │   │   │   └── 1  
  │   │   ├── edge_smoothing  
  │   │   │   └── 0  
  │   │   └── real  
  │   │   └── 0  
  │   └── video_src  
  │   └── Sample_video  
  ├── Saved_model  
  └── Train  
      ├── Pretraining  
      └── Training  
      └── Sample_cartoon

## How to train models

---

If you want to train models, save the cartoon image set in cartoon_images directory.  
Open the Train.py and edit the parameters at line 20-48. At this time, animation_name in line 20 must be same with directory name that the cartoon images are saved.
