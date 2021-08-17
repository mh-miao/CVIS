from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import torch
import math
from PIL import Image
import cv2
import random
from torchvision import transforms


class Part_Inpainting_Dataset(Dataset):

    def __init__(self, img_dir, label_dir, mask_dir, mask_rate=0.5, transform=transforms.Compose([
        #  transforms.Resize(128, 128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.part_bboxes = {
            # 0: [387, 80, 602, 388], 
            # 1: [375, 983, 589, 1292], 
            # 2: [656, 294, 760, 397], 
            # 3: [641, 968, 746, 1071], 
            4: [812, 311, 1488, 458],
            5: [812, 917, 1487, 1062],
            6: [689, 35, 1605, 257],
            7: [667, 1125, 1583, 1348],
#             8: [1626, 37, 1841, 345],
            # 9: [1606, 1048, 1819, 1357],
            10: [116, 374, 306, 969],
            11: [328, 471, 624, 902],
            12: [649, 489, 903, 875],
            13: [946, 535, 1305, 844],
            14: [1334, 506, 1495, 862],
            15: [1533, 424, 1768, 957],
            16: [1783, 337, 1919, 1049],
            # 17: [506, 1378, 1681, 2028]
        }
        self.mask_rate = mask_rate
        self.textures_names = [f for f in os.listdir(img_dir) if 'png' in f]

    def get_label(self, name):
        with open(os.path.join(self.label_dir, name)) as f:
            line = f.readline()

            label = []
            mask_ids = []
            none_ids = []
            index = 0
            for i, x in enumerate(line.split()):
                if i in [0, 1, 2, 3, 8, 9, 17]: continue
                if int(x) == 2:
                    if random.randint(0, 10) < self.mask_rate * 10:
                        label.append(3)
                        mask_ids.append(index)
                    else:
                        label.append(int(x))

                else:
                    if int(x) == 0:
                        none_ids.append(index)
                    label.append(int(x))
                index += 1
            label = torch.tensor(label).long()
            return label

    def process_part(self, texture_img, bbox, is_mask=False, process_raw=True):
        crop_box = (bbox[1], bbox[0], bbox[3], bbox[2])
        part_img = texture_img.crop(crop_box)
        part_img = part_img.resize((128, 128))
        if process_raw:
            part_img = self.transform(part_img)
        else:
            transform = transforms.ToTensor()
            part_img = transform(part_img)
    
        if is_mask: part_img[:, :, :] = 0
        return part_img

    def __len__(self):
        return len(self.textures_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.img_dir, self.textures_names[idx])
        mask_path = os.path.join(self.mask_dir, self.textures_names[idx])

        texture_img = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("RGB")
        label = self.get_label(self.textures_names[idx].replace('png', 'txt'))

        part_img4 = self.process_part(texture_img, self.part_bboxes[4],
                                      is_mask=True if label[0] == 0 or label[0] == 3 else False)
        part_img5 = self.process_part(texture_img, self.part_bboxes[5],
                                      is_mask=True if label[1] == 0 or label[1] == 3 else False)
        part_img6 = self.process_part(texture_img, self.part_bboxes[6],
                                      is_mask=True if label[2] == 0 or label[2] == 3 else False)
        part_img7 = self.process_part(texture_img, self.part_bboxes[7],
                                      is_mask=True if label[3] == 0 or label[3] == 3 else False)
        part_img10 = self.process_part(texture_img, self.part_bboxes[10],
                                      is_mask=True if label[4] == 0 or label[4] == 3 else False)
        part_img11 = self.process_part(texture_img, self.part_bboxes[11],
                                       is_mask=True if label[5] == 0 or label[5] == 3 else False)
        part_img12 = self.process_part(texture_img, self.part_bboxes[12],
                                       is_mask=True if label[6] == 0 or label[6] == 3 else False)
        part_img13 = self.process_part(texture_img, self.part_bboxes[13],
                                       is_mask=True if label[7] == 0 or label[7] == 3 else False)
        part_img14 = self.process_part(texture_img, self.part_bboxes[14],
                                       is_mask=True if label[8] == 0 or label[8] == 3 else False)
        part_img15 = self.process_part(texture_img, self.part_bboxes[15],
                                       is_mask=True if label[9] == 0 or label[9] == 3 else False)
        part_img16 = self.process_part(texture_img, self.part_bboxes[16],
                                       is_mask=True if label[10] == 0 or label[10] == 3 else False)

        gt_part_img4 = self.process_part(texture_img, self.part_bboxes[4],
                                         is_mask=True if label[0] == 0 else False)
        gt_part_img5 = self.process_part(texture_img, self.part_bboxes[5],
                                         is_mask=True if label[1] == 0 else False)
        gt_part_img6 = self.process_part(texture_img, self.part_bboxes[6],
                                         is_mask=True if label[2] == 0 else False)
        gt_part_img7 = self.process_part(texture_img, self.part_bboxes[7],
                                         is_mask=True if label[3] == 0 else False)
        gt_part_img10 = self.process_part(texture_img, self.part_bboxes[10],
                                         is_mask=True if label[4] == 0 else False)
        gt_part_img11 = self.process_part(texture_img, self.part_bboxes[11],
                                          is_mask=True if label[5] == 0 else False)
        gt_part_img12 = self.process_part(texture_img, self.part_bboxes[12],
                                          is_mask=True if label[6] == 0 else False)
        gt_part_img13 = self.process_part(texture_img, self.part_bboxes[13],
                                          is_mask=True if label[7] == 0 else False)
        gt_part_img14 = self.process_part(texture_img, self.part_bboxes[14],
                                          is_mask=True if label[8] == 0 else False)
        gt_part_img15 = self.process_part(texture_img, self.part_bboxes[15],
                                          is_mask=True if label[9] == 0 else False)
        gt_part_img16 = self.process_part(texture_img, self.part_bboxes[16],
                                          is_mask=True if label[10] == 0 else False)

        gt_mask_img4 = self.process_part(mask_img, self.part_bboxes[4],
                                         is_mask=True if label[0] == 0 else False, process_raw=False)
        gt_mask_img5 = self.process_part(mask_img, self.part_bboxes[5],
                                         is_mask=True if label[1] == 0 else False, process_raw=False)
        gt_mask_img6 = self.process_part(mask_img, self.part_bboxes[6],
                                         is_mask=True if label[2] == 0 else False, process_raw=False)
        gt_mask_img7 = self.process_part(mask_img, self.part_bboxes[7],
                                         is_mask=True if label[3] == 0 else False, process_raw=False)
        gt_mask_img10 = self.process_part(mask_img, self.part_bboxes[10],
                                         is_mask=True if label[4] == 0 else False, process_raw=False)
        gt_mask_img11 = self.process_part(mask_img, self.part_bboxes[11],
                                         is_mask=True if label[5] == 0 else False, process_raw=False)
        gt_mask_img12 = self.process_part(mask_img, self.part_bboxes[12],
                                         is_mask=True if label[6] == 0 else False, process_raw=False)
        gt_mask_img13 = self.process_part(mask_img, self.part_bboxes[13],
                                         is_mask=True if label[7] == 0 else False, process_raw=False)
        gt_mask_img14 = self.process_part(mask_img, self.part_bboxes[14],
                                         is_mask=True if label[8] == 0 else False, process_raw=False)
        gt_mask_img15 = self.process_part(mask_img, self.part_bboxes[15],
                                         is_mask=True if label[9] == 0 else False, process_raw=False)
        gt_mask_img16 = self.process_part(mask_img, self.part_bboxes[16],
                                         is_mask=True if label[10] == 0 else False, process_raw=False)      
    
        part_img4[gt_mask_img4 == 0] = 0
        part_img5[gt_mask_img5 == 0] = 0
        part_img6[gt_mask_img6 == 0] = 0
        part_img7[gt_mask_img7 == 0] = 0
        part_img10[gt_mask_img10 == 0] = 0
        part_img11[gt_mask_img11 == 0] = 0
        part_img12[gt_mask_img12 == 0] = 0
        part_img13[gt_mask_img13 == 0] = 0
        part_img14[gt_mask_img14 == 0] = 0
        part_img15[gt_mask_img15 == 0] = 0
        part_img16[gt_mask_img16 == 0] = 0

        return part_img4, part_img5, part_img6, part_img7, part_img10, part_img11, part_img12, part_img13, part_img14, part_img15, part_img16, gt_part_img4, gt_part_img5, gt_part_img6, gt_part_img7, gt_part_img10, gt_part_img11, gt_part_img12, gt_part_img13, gt_part_img14, gt_part_img15, gt_part_img16, gt_mask_img4, gt_mask_img5, gt_mask_img6, gt_mask_img7, gt_mask_img10, gt_mask_img11, gt_mask_img12, gt_mask_img13, gt_mask_img14, gt_mask_img15, gt_mask_img16, label
