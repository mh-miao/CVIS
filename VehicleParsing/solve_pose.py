import pickle
from posixpath import dirname
import cv2
import os
import numpy as np
import math
from pose_sovler_util import pnp_solve, get_xyz_with_given_ry, get_camera_mat,get_rotation_mat, get_model_bounding_box, project_bounding2img
from tqdm import tqdm
from multiprocessing import Process
import random
from densepose.structures import DensePoseResult
import json
import argparse


# x_av, x_std = -0.2189843430163644, 1.4939653358889609
# y_av, y_std = -0.8757352401750409, 0.414332428513173
# z_av, z_std = 0.18008593070005635, 0.5891895773724959
# x_av, x_std = 0.10430685350865337, 0.6010550595020349
# y_av, y_std = 0.09188004204075403, 0.40156067354528396
# z_av, z_std = -0.6366530978820607, 1.5754367130110658
x_av, x_std = 0.015843193737290352, 0.6315879536684835
y_av, y_std = 0.16571132807223296, 0.41599287756036196
z_av, z_std = -0.4589733665950008, 1.5648961244313782
l_av, l_std =  4832.107224634809, 418.2205265935967
w_av, w_std = 2078.2733468579127, 141.02850195946823
h_av, h_std = 1590.6495009377395, 139.03525865033637

def get_color():
	color_table = [
		[60, 20, 220],
		[138, 43, 226],
		[72, 61, 139],
		[0, 0, 205],
		[0, 0, 128],
		[0, 139, 139],
		[0, 128, 0],
		[128, 128, 0],
		[139, 69, 19],
		[254, 0, 255],
		[0, 0, 255],
	]

	return color_table[random.randint(0, len(color_table)-1)]

def get_instance_mask(img, bbox_xyxy, part_xyz):
    mask = part_xyz[0, :, :] > 0
    mask_res = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask_res[int(bbox_xyxy[1]):int(bbox_xyxy[1]) + mask.shape[0], int(bbox_xyxy[0]):int(bbox_xyxy[0]) + mask.shape[1]] = mask
    color = get_color()
    import cv2
    img[:,:,0][mask_res > 0] = color[0] * 0.6 + img[:,:,0][mask_res > 0] * 0.4
    img[:,:,1][mask_res > 0] = color[1] * 0.6 + img[:,:,1][mask_res > 0] * 0.4
    img[:,:,2][mask_res > 0] = color[2] * 0.6 + img[:,:,2][mask_res > 0] * 0.4
    mask_res = np.array(mask_res * 255, np.uint8)
    contours, _ = cv2.findContours(mask_res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,(255,255,255),1)
    return contours, mask_res

def get_image_id(gt_file, img_name):
    with open(gt_file, 'r') as f:
        gt_json = json.load(f)
    images = gt_json['images']
    for img in images:
        if img_name == img['file_name']:
            return img['id']
    return -1


def sovle(data, dir_name, bool_pose_est):
    # for image_id in range(len(data)):
    for image_id in tqdm(range(len(data))):
        print(data[image_id]['pred_boxes_XYXY'])
        target_num = len(data[image_id]['pred_boxes_XYXY'])
        file_name = os.path.basename(data[image_id]['file_name'])
        if 'east' in file_name:
            camera_mat = np.array([[2396.462, 0, 962.126],
                                [0, 2384.34, 578.351],
                                [0, 0, 1]])
        elif 'west' in file_name:
            camera_mat = np.array([[2427.511, 0, 977.600],
                                [0, 2409.163, 558.169],
                                [0, 0, 1]])
        elif 'north' in file_name:
            camera_mat = np.array([[2427.716, 0, 884.505],
                                [0, 2414.880, 555.581],
                                [0, 0, 1]])
        else:
            camera_mat = np.array([[2389.7362, 0, 976.911],
                                [0, 2376.479, 564.022],
                                [0, 0, 1]])
                               
        img = cv2.imread('./img_demo/' + file_name)
        
        print(file_name)

        for instance_id in range(target_num):
            bbox_xyxy = data[image_id]['pred_boxes_XYXY'][instance_id]
            s = data[image_id]['scores'][instance_id]
            shape, part_xyz = data[image_id]['pred_densepose'].results[instance_id]
            if not bool_pose_est:
                if s < 0.99:
                    continue
                _ = get_instance_mask(img, bbox_xyxy, part_xyz)

                # cv2.rectangle(img, (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3])),
                #               (0, 0, 255),
                #               1)
                cv2.putText(img, str(s), (int(bbox_xyxy[0]), int(bbox_xyxy[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                part_img = part_xyz[0, :, :]
                lx = part_xyz[1, :, :]
                ly = part_xyz[2, :, :]
                lz = part_xyz[3, :, :]
                l, w, h = part_xyz[4, 0:3, 0]
                uv = np.argwhere(part_img > 0)
                instance_mask = part_img > 0
                uv[:, [0, 1]] = uv[:, [1, 0]]
                uv[:, 0] += int(bbox_xyxy[0])
                uv[:, 1] += int(bbox_xyxy[1])
                point3d = np.array([
                    lx[instance_mask] * x_std + x_av, ly[instance_mask] * y_std + y_av, lz[instance_mask] * z_std + z_av
                ]).T
                

                # res, ry = get_xyz_with_given_ry(camera_mat, uv, point3d, 5)
                # [a, b, c, x, y, z] = 0, ry, 0, res[0][0], res[1][0], res[2][0]
                try:
                    projection_matrix, rt, pose = pnp_solve(camera_mat, uv, point3d, 5)
                except:
                    continue

                [a, b, c, x, y, z] = pose
                if l != 0:
                    width = (l * l_std + l_av) / 1000
                    length = (w * w_std + w_av) / 1000
                    height = (h * h_std + h_av) / 1000
                else:
                    width = np.max(point3d.T[2, :]) - np.min(point3d.T[2, :])
                    height = np.max(point3d.T[1, :]) - np.min(point3d.T[1, :])
                    length = np.max(point3d.T[0, :]) - np.min(point3d.T[0, :])

                ########## vis poseres ############
                if s < 0.8:
                    continue
                new_point_3d = point3d.T
                E = np.ones((1, new_point_3d.shape[1]))
                new_point_3d = np.vstack((new_point_3d, E))
                new_point_3d = np.dot(projection_matrix, new_point_3d)
                u = new_point_3d[0, :] / new_point_3d[2, :]
                v = new_point_3d[1, :] / new_point_3d[2, :]
                color = get_color()
                for i, j in zip(u, v):
                    if random.randint(0, 100) < 50:
                        continue
                    if i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]:
                        continue
                    img[int(j)][int(i)][0] = color[2]
                    img[int(j)][int(i)][1] = color[1]
                    img[int(j)][int(i)][2] = color[0]

                best_rt = np.hstack((get_rotation_mat(a, b, c), np.array([x, y, z]).reshape(3, 1)))
                best_projection_matrix = np.dot(camera_mat, best_rt)
                bbox3d = get_model_bounding_box(width, height, length)
                project_bounding2img(img, bbox3d, projection_matrix)
            if bool_pose_est:
                cv2.imwrite(os.path.join(dir_name, 'parsing_3d' ,file_name), img)
            else:
                cv2.imwrite(os.path.join(dir_name, 'parsing_2d' ,file_name), img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', default='./model_output')
    parser.add_argument('--output_dir', default='result')
    parser.add_argument('--pose_est', action='store_true')
    parser.add_argument('--img_path', default='./img_demo/')
    parser.add_argument('--gt_path_2d', default='./test.json')

    opt = parser.parse_args()
    print(opt)
    if opt.pose_est:
        file_name = os.path.join(opt.file_dir, 'parsing3d.pkl')
    else:
        file_name = os.path.join(opt.file_dir, 'parsing2d.pkl')

    output_dir = opt.output_dir
    ff = open(file_name, 'rb')
    data = pickle.load(ff)
    print(data)
    print(len(data))
    num_of_worker = 1
    num_per_worker = len(data) // num_of_worker
    if len(data) < num_of_worker:
        num_of_worker = len(data)
        num_per_worker = 12
    processes = []
    
    for i in range(num_of_worker):
        if i == num_of_worker - 1:
            p = Process(target=sovle, args=(
                data[i * num_per_worker:], output_dir, opt.pose_est))
        else:
            p = Process(target=sovle, args=(
                data[i * num_per_worker:(i + 1) * num_per_worker], output_dir, opt.pose_est))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
