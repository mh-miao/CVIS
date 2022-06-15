import pickle
import cv2
import os
import numpy as np
import math

def get_camera_mat(name, calib_dir):
    camera_mat = np.array([[7.215377000000e+02, 0, 6.095593000000e+02],
                        [0, 7.215377000000e+02, 1.728540000000e+02],
                        [0, 0 , 1 ]])
    with open( os.path.join(calib_dir, name.split('.')[0] + '.txt')) as f:
        line = f.readline()
        item = line.split()
        fx, cx, fy, cy = float(item[1]), float(item[3]), float(item[6]), float(item[7])
        camera_mat = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0 , 1 ]])
    return camera_mat

def read_model(point_cloud_path, read_face = False, read_color = False, read_vt = False, scale = 1.0):
    x = []
    y = []
    z = []
    u = []
    v = []
    rgb = []
    face_index = []
    other_strs = []
    with open(point_cloud_path) as f:
        line = f.readline()

        while line:
            if line[0] == 'v' and line[1] == ' ':
                items = line.split(' ')
                x.append(float(items[1]) * scale)
                y.append(float(items[2]) * scale)
                z.append(float(items[3].replace("\n", "")) * scale)
                if read_color:
                    r = float(items[4])
                    g = float(items[5])
                    b = float(items[6].replace("\n", ""))
                    rgb.append([r, g, b])

            elif line[0] == 'f' and read_face:
                other_strs.append(line)
                item = line.split(' ')
                f1 = int(item[1].split('/')[0]) - 1
                f2 = int(item[2].split('/')[0]) - 1
                f3 = int(item[3].split('/')[0]) - 1
                if len(item) == 5:
                    f4 = int(item[4].split('/')[0]) - 1
                    face_index.append([f1, f2, f3, f4])
                else:
                    face_index.append([f1, f2, f3])
            elif line[0] == 'v' and line[1] == 't':
                other_strs.append(line)
                items = line.split(' ')
                u.append(int((float(items[1])) * 2048))
                v.append(int((1 - float(items[2])) * 2048))
            line = f.readline()

    return np.array([x, y, z]), np.array(face_index), np.array(u), np.array(v), np.array(rgb).T, other_strs

def papare_sim_models(model_dir):
    model_names = [f for f in os.listdir(model_dir) if 'obj' in f]
    model_names.sort()
    pcs, face_indexs, t_us, t_vs = [], [], [], []
    import math
    for name in model_names:
        pc, face_index, t_u, t_v, _, _ = read_model(os.path.join(model_dir, name), read_face=True, read_vt=True)
        pc = np.dot(get_rotation_mat(math.pi/2, 0, 0), pc)
        # pc = np.dot(get_rotation_mat(0, -math.pi/2, 0), pc)
        pcs.append(pc)
        face_indexs.append(face_index)
        t_us.append(t_u)
        t_vs.append(t_v)
    return pcs, face_indexs, t_us, t_vs

def get_rotation_mat(a, b, c):
    rotation = np.zeros((3,3))
    rotation[0][0] = math.cos(c) * math.cos(b)
    rotation[0][1] = -math.sin(c) * math.cos(a) + math.cos(c) * math.sin(b) * math.sin(a)
    rotation[0][2] = math.sin(a) * math.sin(c) + math.cos(c) * math.sin(b) * math.cos(a)
    rotation[1][0] = math.cos(b) * math.sin(c)
    rotation[1][1] = math.cos(c) * math.cos(a) + math.sin(c) * math.sin(b) * math.sin(a)
    rotation[1][2] = -math.sin(a) * math.cos(c) + math.cos(a) * math.sin(b) * math.sin(c)
    rotation[2][0] = -math.sin(b)
    rotation[2][1] = math.cos(b) * math.sin(a)
    rotation[2][2] = math.cos(a) * math.cos(b)
    return rotation

def project_bounding2img(img, bounding,projection_matrix):
    p = np.dot(projection_matrix, bounding)
    v = p[1, :] / p[2, :]
    u = p[0, :] / p[2, :]
    v[v < 0] = 0
    v[v > img.shape[0]] = img.shape[0]
    u[u < 0] = 0
    u[u > img.shape[1]] = img.shape[1]
    min_u = np.min(u)
    min_v = np.min(v)
    color = (0, 255, 0)
    cv2.line(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), color, 2)
    cv2.line(img, (int(u[0]), int(v[0])), (int(u[3]), int(v[3])), color, 2)
    cv2.line(img, (int(u[1]), int(v[1])), (int(u[2]), int(v[2])), color, 2)
    cv2.line(img, (int(u[2]), int(v[2])), (int(u[3]), int(v[3])), color, 2)

    cv2.line(img, (int(u[4]), int(v[4])), (int(u[5]), int(v[5])), color, 2)
    cv2.line(img, (int(u[5]), int(v[5])), (int(u[6]), int(v[6])), color, 2)
    cv2.line(img, (int(u[6]), int(v[6])), (int(u[7]), int(v[7])), color, 2)
    cv2.line(img, (int(u[7]), int(v[7])), (int(u[4]), int(v[4])), color, 2)

    cv2.line(img, (int(u[0]), int(v[0])), (int(u[4]), int(v[4])), color, 2)
    cv2.line(img, (int(u[1]), int(v[1])), (int(u[5]), int(v[5])), color, 2)
    cv2.line(img, (int(u[2]), int(v[2])), (int(u[6]), int(v[6])), color, 2)
    cv2.line(img, (int(u[3]), int(v[3])), (int(u[7]), int(v[7])), color, 2)

    return img

def get_model_bounding_box(w, h, l):
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    ones = [1, 1, 1, 1, 1, 1, 1, 1]
    bouding = np.array([
        x_corners, y_corners, z_corners, ones
    ])

    # bouding = np.array([p1,p2,p3,p4,p5,p6,p7,p8])
    return bouding


def R_Mat_to_Euler(R):
    # float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    if sy > 1e-6:
        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])
    else:
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], sy)
        z = 0

    return x, y, z
def get_xyz_with_given_ry(camera_matrix, point_2d, point_3d, range_step):
    '''

    Args:
        camera_matrix:
        point_2d: n*2
        point_3d: n*3
        ry:

    Returns:

    '''
    min_error = 1e9
    res = None
    ry_res = None
    for ry_rad in range(360//range_step):
        ry = ry_rad * range_step / 180.0 * math.pi
        r1, r2 = math.sin(ry), math.cos(ry)
        num_points = point_2d.shape[0]
        fx, fy, cx, cy = camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]
        uu = (point_2d[:, 0] - cx) / fx
        vv = (point_2d[:, 1] - cy) / fy
        A = np.zeros((num_points * 2, 3))
        b = np.zeros((num_points * 2, 1))
        for i in range(num_points):
            A[i * 2][0] = 1
            A[i * 2 + 1][1] = 1
            A[i * 2][2] = -uu[i]
            A[i * 2 + 1][2] = -vv[i]

            b[i * 2] = -uu[i] * r1 * point_3d[i][0] + uu[i] * r2 * point_3d[i][2] - r2 * point_3d[i][0] - r1 * \
                       point_3d[i][2]
            b[i * 2 + 1] = -vv[i] * r1 * point_3d[i][0] + vv[i] * r2 * point_3d[i][2] - point_3d[i][1]

        x, residuals, rank, s = np.linalg.lstsq(A, b)
        if residuals < min_error:
            min_error = residuals
            res = x
            ry_res = ry
    return res, ry_res

def pnp_solve(camera_matrix, point_2d, point_3d, reprojectionError):
    # print(point_2d.shape)
    # print(point_3d.shape)
    point_2d = np.float32(point_2d)
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(point_3d, point_2d, camera_matrix, np.zeros((5,1)), iterationsCount=1000, reprojectionError=reprojectionError, flags=1)
    # print(len(inliers))
    # _,rvec, tvec = cv2.solvePnP(point_3d, point_2d, camera_matrix, np.zeros((5, 1)))
    pose = [rvec[0][0], rvec[1][0], rvec[2][0] , tvec[0][0], tvec[1][0], tvec[2][0]]
    rvec, jac = cv2.Rodrigues(rvec)
    a, b, c = R_Mat_to_Euler(rvec)
    pose[0] = a
    pose[1] = b
    pose[2] = c
    if pose[5] < 0:
        pose[0] += math.pi
        pose[3] = -pose[3]
        pose[4] = -pose[4]
        pose[5] = -pose[5]

    Rt = np.hstack((rvec, tvec))
    return np.dot(camera_matrix, Rt), Rt, pose