import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

import time


class Part_GCN(nn.Module):

    def __init__(self, nc=3, ngf=4):
        super(Part_GCN, self).__init__()
        self.ngf = ngf
        self.encoder = nn.Sequential(
                    nn.Conv2d(nc, ngf * 8, 4, 2, 1),
                    # nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                )
        self.encoder2 = nn.Sequential(
                    nn.Conv2d(ngf * 8 * 2, ngf * 8 * 2, 4, 2, 1),
                    # nn.BatchNorm2d(ngf * 8 * 2),
                    nn.ReLU(True),
                )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(ngf * 8 * 4, ngf * 8 * 4, 4, 2, 1),
            # nn.BatchNorm2d(ngf * 8 * 4),
            nn.ReLU(True),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(ngf * 8 * 8, ngf * 8 * 8, 4, 2, 1),
            # nn.BatchNorm2d(ngf * 8 * 8),
            nn.ReLU(True),
        )
        self.decode_group = nn.ModuleList([
            nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 * 16, ngf * 8 * 8, 4, 2, 1),
                    # nn.BatchNorm2d(ngf * 8 * 8),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(ngf * 8 * 8, ngf * 8 * 2, 4, 2, 1),
                    # nn.BatchNorm2d(ngf * 8 * 2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1),
                    # nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(ngf * 8, nc, 4, 2, 1),
#                     nn.BatchNorm2d(nc),
#                     nn.Tanh()
                )
            for i in range(11)])

        self.l1 = nn.SmoothL1Loss(reduction='sum')

    def gcn_forword(self, part_imgs, encoder):
        part_imgs = [encoder(part_img) for part_img in part_imgs]
        part_img_aggregate = [torch.unsqueeze(part_img, 1) for part_img in part_imgs]
        # aggregate
        aggregate = torch.cat(part_img_aggregate, 1) # N * part * c * h * w
        aggregate_feature, _ = torch.max(aggregate, 1) # N * c * h * w
        part_imgs = [torch.cat((part_img, aggregate_feature), 1) for part_img in part_imgs]
        # print(part_imgs[0].shape)
        return part_imgs


    def forward(self, part_imgs, gt_part_imgs=None, gt_mask_imgs=None, label=None):
        part_imgs = self.gcn_forword(part_imgs, self.encoder)
        part_imgs = self.gcn_forword(part_imgs, self.encoder2)
        part_imgs = self.gcn_forword(part_imgs, self.encoder3)
        part_imgs = self.gcn_forword(part_imgs, self.encoder4)
        recon_imgs = [decoder(part_img) for part_img, decoder in zip(part_imgs, self.decode_group)]

        recon_imgs = [torch.unsqueeze(recon_img, 1) for recon_img in recon_imgs]
        recon_imgs = torch.cat(recon_imgs, 1)
        if self.training:
            gt_part_imgs = [torch.unsqueeze(gt_part_img, 1) for gt_part_img in gt_part_imgs]
            gt_part_imgs = torch.cat(gt_part_imgs, 1)
            # print(gt_mask_imgs)
            gt_mask_imgs = [torch.unsqueeze(gt_mask_img, 1) for gt_mask_img in gt_mask_imgs]
            gt_mask_imgs = torch.cat(gt_mask_imgs, 1)

            loss = self.recon_loss(gt_part_imgs, recon_imgs, gt_mask_imgs, label)
            return recon_imgs, loss
        else:
            return recon_imgs


    def recon_loss(self, gt_imgs, recon_imgs, gt_mask_imgs, masks):
        '''
        :param gt_imgs:
        :param recon_imgs:
        :param masks: N * part
        :return:
        '''
        loss = {}
        # print(gt_imgs.shape, recon_imgs.shape, masks.shape, gt_mask_imgs.shape)
        # print(torch.sum(masks==3))
        # recon_loss = self.l1(gt_imgs[masks>1, :, :, :], recon_imgs[masks>1, :, :, :])/(torch.sum(masks>1)*128*128)
        recon_loss = self.l1(gt_imgs[gt_mask_imgs > 0], recon_imgs[gt_mask_imgs > 0])/(torch.sum(gt_mask_imgs > 0) / 3)
        loss['recon_loss'] = recon_loss
        return loss

# model = Part_GCN()
# model.eval()
# input = torch.zeros((2, 18, 3, 256, 256))
# part_img4 = torch.zeros((2, 3, 64, 64))
# part_img5 = torch.zeros((2, 3, 64, 64))
# part_img6 = torch.zeros((2, 3, 64, 64))
# part_img7 = torch.zeros((2, 3, 64, 64))
# part_img8 = torch.zeros((2, 3, 64, 64))
# part_img11 = torch.zeros((2, 3, 64, 64))
# part_img12 = torch.zeros((2, 3, 64, 64))
# part_img13 = torch.zeros((2, 3, 64, 64))
# part_img14 = torch.zeros((2, 3, 64, 64))
# part_img15 = torch.zeros((2, 3, 64, 64))
# part_img16 = torch.zeros((2, 3, 64, 64))
# # masks = torch.zeros((2, 18), dtype=torch.long)
# recon = model(part_img4,
#                 part_img5,
#                 part_img6,
#                 part_img7,
#                 part_img8,
#                 part_img11,
#                 part_img12,
#                 part_img13,
#                 part_img14,
#                 part_img15,
#                 part_img16)




if __name__ == '__main__':
    base_path = os.path.abspath('..')
    
    texture_dir = os.path.join(base_path, 'Datasets', '00_texture_init', 'symmetry_images')
    texture_label_dir = os.path.join(base_path, 'Datasets', '00_texture_init', 'symmetry_integrity_txt')
    texture_mask_dir = os.path.join(base_path, 'Datasets', '00_texture_init', 'symmetry_mask')
    texture_templete_path = os.path.join(base_path, 'Datasets', '00_texture_init', 'Template18_new.PNG')
    texture_output_dir = os.path.join(base_path, 'Datasets', '01_texture_inpainting', 'images')
    
    from fill_utils import get_part_mask1, get_missing_region
    texture_mask_templete_dict = get_part_mask1(texture_templete_path)
    
    transform = transforms.Compose([
        #  transforms.Resize(128, 128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    part_bboxes = {
        # 0: [387, 80, 602, 388],
        # 1: [375, 983, 589, 1292],
        # 2: [656, 294, 760, 397],
        # 3: [641, 968, 746, 1071],
        4: [812, 311, 1488, 458],
        5: [812, 917, 1487, 1062],
        6: [689, 35, 1605, 257],
        7: [667, 1125, 1583, 1348],
        # 8: [1626, 37, 1841, 345],
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
    part_bboxes2 = [v for k, v in part_bboxes.items()]
    
    texture_names = [f.split('.')[0] for f in os.listdir(texture_dir) if 'png' in f]
    model = Part_GCN()
    static_dict = torch.load('gcn_model.pth', map_location='cpu')
    model.load_state_dict((static_dict['model']))
    model.to('cuda')
    model.eval()
    
    for texture_name in texture_names:
        # texture_name = '171206_034808931_Camera_5_0'

        label = []
        with open(os.path.join(texture_label_dir, texture_name + '.txt')) as f:
            line = f.readline()
            l = [int(x) for x in line.split()]
            for i in [4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]:
                label.append(l[i])

        texture_img = Image.open(os.path.join(texture_dir, texture_name + '.png')).convert('RGB')
        mask_img = Image.open(os.path.join(texture_mask_dir, texture_name + '.png')).convert("RGB")
        texture_mask = cv2.imread(os.path.join(texture_mask_dir, texture_name + '.png'))
        texture_show = cv2.imread(os.path.join(texture_dir, texture_name + '.png'))
        # cv2.imwrite(os.path.join(texture_output_dir, texture_name + '.png'), texture_show)
        def process_part(texture_img, bbox, is_mask=False, process_raw=True):
            crop_box = (bbox[1], bbox[0], bbox[3], bbox[2])
            part_img = texture_img.crop(crop_box)
            part_img = part_img.resize((128, 128))
            if process_raw:
                part_img = transform(part_img)
            else:
                transform_mask = transforms.ToTensor()
                part_img = transform_mask(part_img)
            if is_mask: part_img[:, :, :] = 0
            return part_img
        part_img4 = process_part(texture_img, part_bboxes[4],
                                 is_mask=True if label[0] == 0 or label[0] == 3 else False)
        part_img5 = process_part(texture_img, part_bboxes[5],
                                 is_mask=True if label[1] == 0 or label[1] == 3 else False)
        part_img6 = process_part(texture_img, part_bboxes[6],
                                 is_mask=True if label[2] == 0 or label[2] == 3 else False)
        part_img7 = process_part(texture_img, part_bboxes[7],
                                 is_mask=True if label[3] == 0 or label[3] == 3 else False)
        part_img10 = process_part(texture_img, part_bboxes[10],
                                  is_mask=True if label[4] == 0 or label[4] == 3 else False)
        part_img11 = process_part(texture_img, part_bboxes[11],
                                  is_mask=True if label[5] == 0 or label[5] == 3 else False)
        part_img12 = process_part(texture_img, part_bboxes[12],
                                  is_mask=True if label[6] == 0 or label[6] == 3 else False)
        part_img13 = process_part(texture_img, part_bboxes[13],
                                  is_mask=True if label[7] == 0 or label[7] == 3 else False)
        part_img14 = process_part(texture_img, part_bboxes[14],
                                  is_mask=True if label[8] == 0 or label[8] == 3 else False)
        part_img15 = process_part(texture_img, part_bboxes[15],
                                  is_mask=True if label[9] == 0 or label[9] == 3 else False)
        part_img16 = process_part(texture_img, part_bboxes[16],
                                  is_mask=True if label[10] == 0 or label[10] == 3 else False)
        
        gt_mask_img4 = process_part(mask_img, part_bboxes[4],
                                         is_mask=True if label[0] == 0 else False, process_raw=False)
        gt_mask_img5 = process_part(mask_img, part_bboxes[5],
                                         is_mask=True if label[1] == 0 else False, process_raw=False)
        gt_mask_img6 = process_part(mask_img, part_bboxes[6],
                                         is_mask=True if label[2] == 0 else False, process_raw=False)
        gt_mask_img7 = process_part(mask_img, part_bboxes[7],
                                         is_mask=True if label[3] == 0 else False, process_raw=False)
        gt_mask_img10 = process_part(mask_img, part_bboxes[10],
                                         is_mask=True if label[4] == 0 else False, process_raw=False)
        gt_mask_img11 = process_part(mask_img, part_bboxes[11],
                                         is_mask=True if label[5] == 0 else False, process_raw=False)
        gt_mask_img12 = process_part(mask_img, part_bboxes[12],
                                         is_mask=True if label[6] == 0 else False, process_raw=False)
        gt_mask_img13 = process_part(mask_img, part_bboxes[13],
                                         is_mask=True if label[7] == 0 else False, process_raw=False)
        gt_mask_img14 = process_part(mask_img, part_bboxes[14],
                                         is_mask=True if label[8] == 0 else False, process_raw=False)
        gt_mask_img15 = process_part(mask_img, part_bboxes[15],
                                         is_mask=True if label[9] == 0 else False, process_raw=False)
        gt_mask_img16 = process_part(mask_img, part_bboxes[16],
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
        
        part_imgs = [part_img4, part_img5, part_img6, part_img7, part_img10, part_img11, part_img12, part_img13,
                     part_img14, part_img15, part_img16]
        part_imgs = [torch.unsqueeze(part, 0).to('cuda') for part in part_imgs]
        # part_imgs = part_imgs.to('cuda')
        s = time.time()
        recon = model(part_imgs)
        e = time.time()
        
        for i, l in enumerate(label):
            res = np.ones((128, 128, 3), dtype=np.uint8)
            recon_part = recon[0][i].cpu().detach().numpy()
            recon_part = recon_part.transpose(1, 2, 0)
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            res[:, :, 2] = (recon_part[:, :, 0] * 0.229 + 0.485) * 255
            res[:, :, 1] = (recon_part[:, :, 1] * 0.224 + 0.456) * 255
            res[:, :, 0] = (recon_part[:, :, 2] * 0.225 + 0.406) * 255
            # print(np.max(res[:, :, 0]))
            # print(np.max(res[:, :, 1]))
            # print(np.max(res[:, :, 2]))
            bbox = part_bboxes2[i]
            res = cv2.resize(res, (bbox[3] - bbox[1], bbox[2] - bbox[0]))
            part_missing = get_missing_region(texture_mask, texture_mask_templete_dict[i], bbox)
            
            texture_show[bbox[0]:bbox[2], bbox[1]:bbox[3], :][part_missing] = res[part_missing]
            # cv2.imwrite(str(i) + '.png', res)
        if not os.path.exists(texture_output_dir):
            os.makedirs(texture_output_dir)
        cv2.imwrite(os.path.join(texture_output_dir, texture_name + '.png'), texture_show)