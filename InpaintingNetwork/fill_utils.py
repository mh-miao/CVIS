import cv2
import numpy as np

# part rgb color in templete
def get_texture_part_color():
    part0 = [100, 111, 156]
    part1 = [5, 28, 126]
    part2 = [188, 110, 83]
    part3 = [75, 13, 159]
    part4 = [8, 78, 183]
    part5 = [216, 166, 255]
    part6 = [113, 165, 0]
    part7 = [229, 114, 84]
    part8 = [140, 60, 39]
    part9 = [112, 252, 57]

    part10 = [247, 77, 149]
    part11 = [32, 148, 9]
    part12 = [166, 104, 6]
    part13 = [7, 212, 133]
    part14 = [1, 251, 1]
    part15 = [2, 2, 188]
    part16 = [219, 251, 1]
    part17 = [96, 94, 92]

    return [part0, part1, part2, part3, part4, part5, part6, part7, part8, part9, part10, part11, part12, part13, part14, part15, part16, part17]

# part bbox in templete
def get_part_patch_box(texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    parts_color = get_texture_part_color()
    texture_bbox_dict = {}
    for id, color in enumerate(parts_color):
        area_uv = np.argwhere(texture_map[:, :, 2] == color[0])
        min_v = np.min(area_uv[:, 0])
        min_u = np.min(area_uv[:, 1])
        max_v = np.max(area_uv[:, 0])
        max_u = np.max(area_uv[:, 1])
        texture_bbox_dict[id] = [min_v, min_u, max_v, max_u]
    return texture_bbox_dict

# part mask in templete
def get_part_mask(texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    parts_color = get_texture_part_color()
    texture_mask_dict = {}
    for i, color in enumerate(parts_color):
        texture_mask_dict[i] = texture_map[:, :, 2] == color[0]
    return texture_mask_dict

def get_part_mask1(texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    parts_color = get_texture_part_color()
    texture_mask_dict = {}
    index = 0
    for i, color in enumerate(parts_color):
        if i in [4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]:
            texture_mask_dict[index] = texture_map[:, :, 2] == color[0]
            index = index + 1
    return texture_mask_dict

def get_missing_region(mask_img, part_mask, bbox):
    part_mask_exist = mask_img[bbox[0]:bbox[2], bbox[1]:bbox[3], 0] != 0
    part_mask_missing = (part_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] != 0) & (part_mask_exist == 0)

    return part_mask_missing


