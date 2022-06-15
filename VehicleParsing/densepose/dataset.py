# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

_URL_PREFIX = "https://dl.fbaipublicfiles.com/densepose/data/"


def get_densepose_metadata():
    meta = {
        "thing_classes": ["car"],
        "densepose_transform_src": _URL_PREFIX + "UV_symmetry_transforms.mat",
        "densepose_smpl_subdiv": _URL_PREFIX + "SMPL_subdiv.mat",
        "densepose_smpl_subdiv_transform": _URL_PREFIX + "SMPL_SUBDIV_TRANSFORM.mat",
    }
    return meta

dataset_root_path = os.path.abspath(__file__).split('VehicleParsing')[0]


SPLITS = {
    "densepose_coco_2014_train": ("coco/train2014", "coco/annotations/densepose_train2014.json"),
    "densepose_coco_2014_minival": ("coco/val2014", "coco/annotations/densepose_minival2014.json"),
    "densepose_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/densepose_minival2014_100.json",
    ),
    "densepose_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/densepose_valminusminival2014.json",
    ),
    "kitti_lxyz": (
        "/home/beta/SKP/data/Kitti3D/train/image",
        "/home/beta/CVPR21/local_xyz_generation/kitti_dense_localxyz_sample_test.json",
    ),
    "cross_lxyz": (
        "/media/vrlab/新加卷/CVPR2021/" + "Datasets/02_Part_inpainting_shadow/images",
        "/media/vrlab/新加卷/CVPR2021/" + "Datasets/02_Part_inpainting_shadow/local_xyz_json_1k/crossroad_dense_localxyz_norm.json",
    ),
    "cross_lxyz_dimension": (
        dataset_root_path + "Datasets/02_Part_inpainting_shadow/images",
        dataset_root_path + "Datasets/02_Part_inpainting_shadow/local_xyz_dimension/crossroad_dense_localxyz_dimension_norm.json",
    ),
    "cross_pure": (
        dataset_root_path + "Datasets/05_Pure/images",
        dataset_root_path + "Datasets/02_Part_inpainting_shadow/local_xyz_json_1k/crossroad_dense_localxyz_norm.json",
    ),
    "cross_knn": (
        dataset_root_path + "Datasets/06_KNN/images",
        dataset_root_path + "Datasets/02_Part_inpainting_shadow/local_xyz_json_1k/crossroad_dense_localxyz_norm.json",
    ),
    "cross_inpainting": (
        dataset_root_path + "Datasets/07_Image_Inpainting/images",
        dataset_root_path + "Datasets/02_Part_inpainting_shadow/local_xyz_json_1k/crossroad_dense_localxyz_norm.json",
    ),
    "cross_woshadow": (
        dataset_root_path + "Datasets/08_Part_inpainting_bg/images",
        dataset_root_path + "Datasets/02_Part_inpainting_shadow/local_xyz_json_1k/crossroad_dense_localxyz_norm.json",
    ),
}

DENSEPOSE_KEYS = ["dp_x", "dp_y", "dp_I", "dp_lx", "dp_ly", "dp_lz", "dp_masks"]
DENSEPOSE_DIM_KEYS = ["dp_x", "dp_y", "dp_I", "dp_lx", "dp_ly", "dp_lz", "dp_masks", 'dimension']

for key, (image_root, json_file) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.
    # json_file = os.path.join("datasets", json_file)
    # image_root = os.path.join("datasets", image_root)

    DatasetCatalog.register(
        key,
        lambda key=key, json_file=json_file, image_root=image_root: load_coco_json(
            json_file, image_root, key, extra_annotation_keys=DENSEPOSE_DIM_KEYS
        ),
    )

    MetadataCatalog.get(key).set(
        json_file=json_file, image_root=image_root, **get_densepose_metadata()
    )
