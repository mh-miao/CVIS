# Robust 2D/3D Vehicle Parsing in Arbitrary Camera Views for CVIS

## Abstract
We present a novel approach to robustly detect and perceive vehicles in different camera views as part of a cooperative vehicle-infrastructure system (CVIS). Our formulation is designed for arbitrary camera views and makes no assumptions about intrinsic or extrinsic parameters. First, to deal with multi-view data scarcity, we propose a part-assisted novel view synthesis algorithm for data augmentation. We train a part-based texture inpainting network in a self-supervised manner. Then we render the textured model into the background image with the target 6-DoF pose. Second, to handle various camera parameters, we present a new method that produces dense mappings between image pixels and 3D points to perform robust 2D/3D vehicle parsing. Third, we build the first CVIS dataset for benchmarking, which annotates more than 1540 images (14017 instances) from real-world traffic scenarios. We combine these novel algorithms and datasets to develop a robust approach for 2D/3D vehicle parsing for CVIS. In practice, our approach outperforms SOTA methods on 2D detection, instance segmentation, and 6-DoF pose estimation by 3.8%, 4.3%, and 2.9%, respectively.
## Texture Inpainting Network
```
cd InpaintingNetwork
```
### Requirements
- python3.6, pytorch1.4.0, torchvision0.5.0
- opencv-python
### Training
```
python train.py
```
### Inpainting
```
python partgcn.py
```
### Refinement
```
python refine_inpainting.py
```
The pretrained model can be downloaded at [Google Drive](https://drive.google.com/file/d/1GhF1bSdcDPwG4wme8RJsDL5nZyFrkxIu/view?usp=sharing)
## Contact
For questions regarding our work, please contact the authors (miaohui_@buaa.edu.cn).
