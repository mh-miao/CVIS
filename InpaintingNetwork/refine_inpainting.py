import cv2
import os
import numpy as np
from multiprocessing import Process
from fill_utils import get_part_patch_box, get_part_mask

def get_avg_color(texture_img, parts_mask):
    mask_8 = parts_mask[6]
    avg_b = np.average(texture_img[:, :, 0][mask_8])
    avg_g = np.average(texture_img[:, :, 1][mask_8])
    avg_r = np.average(texture_img[:, :, 2][mask_8])

    return avg_b, avg_g, avg_r

def fill_red_region(texture_img, parts_bbox, parts_mask):
    [min_v_2, min_u_2, max_v_2, max_u_2] = parts_bbox[2]
    [min_v_3, min_u_3, max_v_3, max_u_3] = parts_bbox[3]

    mask_red_2 = texture_img[:, :, 2][min_v_2 : max_v_2 + 1, min_u_2 : max_u_2 + 1] > 100
    mask_red_3 = texture_img[:, :, 2][min_v_3 : max_v_3 + 1, min_u_3 : max_u_3 + 1] > 100

    avg_b, avg_g, avg_r = get_avg_color(texture_img, parts_mask)

    mask = np.zeros((2048, 2048))
    mask[min_v_2 : max_v_2 + 1, min_u_2 : max_u_2 + 1][mask_red_2] = 1
    mask[min_v_3 : max_v_3 + 1, min_u_3 : max_u_3 + 1][mask_red_3] = 1
    mask = mask == 1

    texture_img[:,:,0][mask] = avg_b
    texture_img[:,:,1][mask] = avg_g
    texture_img[:,:,2][mask] = avg_r
    
    return texture_img

def refine_inpainting_res(texture_list, texture_templete_car, path_dict):
    texture_templete_color_path = path_dict['texture_templete_color_path']
    texture_inpainting_dir = path_dict['texture_inpainting_dir']
    texture_output_dir = path_dict['texture_output_dir']

    parts_bbox = get_part_patch_box(texture_templete_color_path)
    parts_mask = get_part_mask(texture_templete_color_path)

    for texture_file in texture_list:
        print(texture_file)
        texture_path = os.path.join(texture_inpainting_dir, texture_file)
        texture_img = cv2.imread(texture_path)
        for part_id in [0, 1, 2, 3, 8, 9, 17]:
            part_mask = parts_mask[part_id]
            part_bbox = parts_bbox[part_id]
           
            texture_img[:,:,0][part_mask] = texture_templete_car[:,:,0][part_mask]
            texture_img[:,:,1][part_mask] = texture_templete_car[:,:,1][part_mask]
            texture_img[:,:,2][part_mask] = texture_templete_car[:,:,2][part_mask]
        
        texture_img = fill_red_region(texture_img, parts_bbox, parts_mask)
        if not os.path.exists(texture_output_dir):
            os.makedirs(texture_output_dir)
        cv2.imwrite(os.path.join(texture_output_dir, texture_file), texture_img)

if __name__ == "__main__":
    base_path = os.path.abspath('..')
    texture_templete_car_path = os.path.join(base_path, 'Datasets', '00_texture_init', '11.png')
    texture_templete_color_path = os.path.join(base_path, 'Datasets', '00_texture_init', 'Template18_new.PNG')
    
    texture_inpainting_dir = os.path.join(base_path, 'Datasets', '01_texture_inpainting', 'images')
    texture_output_dir = os.path.join(base_path, 'Datasets', '01_texture_inpainting', 'images_refine')
    texture_templete_car = cv2.imread(texture_templete_car_path)

    texture_inpainting_list = os.listdir(texture_inpainting_dir)
    # texture_inpainting_list = ['171206_034559609_Camera_5_normal_0_res.png']
    print(len(texture_inpainting_list))

    path_dict = {'texture_output_dir':texture_output_dir, 'texture_inpainting_dir':texture_inpainting_dir, 'texture_templete_car_path':texture_templete_car_path, 'texture_templete_color_path':texture_templete_color_path}

    num_of_worker = 20
    num_per_worker = len(texture_inpainting_list) // num_of_worker
    processes = []
    for i in range(num_of_worker):
        if i == num_of_worker - 1:
            p = Process(target=refine_inpainting_res, args=(texture_inpainting_list[i * num_per_worker:], texture_templete_car, path_dict))
        else:
            p = Process(target=refine_inpainting_res, args=(texture_inpainting_list[i * num_per_worker:(i + 1) * num_per_worker], texture_templete_car, path_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
