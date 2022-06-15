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

## Vehicle Parsing Network
This code is based on detectron2(densepose) but modified to realize canonical point regression module and dimension[w, h, l] regression module.
```
cd VehicleParsing
```
### Requirements
- python3.6, cuda10.0
- detectron2 0.1.1
### Training
```
mkdir -p ImageNetPretrained/MSRA
```
Download the pretrained model at [URL1](https://drive.google.com/file/d/10rggOtosWStS9WzD4ydbWB5X8JgG9WPa/view?usp=sharing) [URL2](https://drive.google.com/file/d/1HYj6IaAAgsVZcgxeheBqndsjgpvXaBEL/view?usp=sharing) and put these model in ./ImageNetPretrained/MSRA 


```
python train_net.py --config-file configs/***.yaml  --num-gpu N
```
### Parsing
- 2D Parsing
```
mkdir model

python apply_net.py dump configs/rcnn_R_101_FPN_DL_s1x.yaml ./model/parsing2d.pth ./img_demo --output model_output/parsing2d.pkl -v

mkdir -p result/parsing_2d

python solve_pose.py
```
The pretrained model can be downloaded at [Google Drive](https://drive.google.com/file/d/1oP5Sj2BV_RsYNMVoe9-WfwD5lGBxRsw8/view?usp=sharing) and put the model in ./model
- 3D Parsing
```
python apply_net.py dump configs/rcnn_R_50_FPN_DL_s1x.yaml ./model/parsing3d.pth ./img_demo --output model_output/parsing3d.pkl -v

mkdir -p result/parsing_3d

python solve_pose.py --pose_est
```
The pretrained model can be downloaded at [Google Drive](https://drive.google.com/file/d/1wqXmMGUa6281qZUmy0yAX5_Mql5kTCAq/view?usp=sharing) and put the model in ./model
## Dataset
Our data involves privacy, and if you want to require it, please contact us directly.
## Contact
For questions regarding our work, feel free to post here or directly contact the authors (miaohui_@buaa.edu.cn).
