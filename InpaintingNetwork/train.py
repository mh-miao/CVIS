import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
# import transforms as T
import utils
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from inpainting_dataloader import Part_Inpainting_Dataset
import math
import json
from torchvision.transforms import functional as F
from partgcn import Part_GCN


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    sub_itera = 0
    for part_img4, part_img5, part_img6, part_img7, part_img8, part_img11, part_img12, part_img13, part_img14, part_img15, part_img16, gt_part_img4, gt_part_img5, gt_part_img6, gt_part_img7, gt_part_img8, gt_part_img11, gt_part_img12, gt_part_img13, gt_part_img14, gt_part_img15, gt_part_img16, gt_mask_img4, gt_mask_img5, gt_mask_img6, gt_mask_img7, gt_mask_img10, gt_mask_img11, gt_mask_img12, gt_mask_img13, gt_mask_img14, gt_mask_img15, gt_mask_img16, label in metric_logger.log_every(data_loader, print_freq, header):
        part_imgs = [part_img4, part_img5, part_img6, part_img7, part_img8, part_img11, part_img12, part_img13, part_img14, part_img15, part_img16]
        gt_imgs = [gt_part_img4, gt_part_img5, gt_part_img6, gt_part_img7, gt_part_img8, gt_part_img11, gt_part_img12, gt_part_img13, gt_part_img14, gt_part_img15, gt_part_img16]
        gt_mask_imgs = [gt_mask_img4, gt_mask_img5, gt_mask_img6, gt_mask_img7, gt_mask_img10, gt_mask_img11, gt_mask_img12, gt_mask_img13, gt_mask_img14, gt_mask_img15, gt_mask_img16]
        label = label.to(device)
        part_imgs = [part_img.to(device) for part_img in part_imgs]
        gt_imgs = [gt_img.to(device) for gt_img in gt_imgs]
        gt_mask_imgs = [gt_mask_img.to(device) for gt_mask_img in gt_mask_imgs]
        # print(label.shape)
        _, loss_dict = model(part_imgs, gt_imgs, gt_mask_imgs, label)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def main(args):
    base_path = os.path.abspath('..')
    img_dir = os.path.join(base_path, 'Datasets', '00_texture_init', 'symmetry_images')
    label_dir = os.path.join(base_path, 'Datasets', '00_texture_init', 'symmetry_integrity_txt')
    mask_dir = os.path.join(base_path, 'Datasets', '00_texture_init', 'symmetry_mask')
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset = Part_Inpainting_Dataset(img_dir, label_dir, mask_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #  test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
    #  test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, drop_last=True)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1,
    #     sampler=test_sampler, num_workers=args.workers)

    model = Part_GCN()
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # if args.test_only:
    #     confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
    #     print(confmat)
    #     return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
    ]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        # print(confmat)
        if epoch % 10 == 0:
            if args.output_dir:
                utils.mkdir(args.output_dir)
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args
                },
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('-distribute', action="store_true")
    parser.add_argument('--epochs', default=1001, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./gcn/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--min-x', default=-27, type=int, help='print frequency')
    parser.add_argument('--max-x', default=27, type=int, help='print frequency')
    parser.add_argument('--min-z', default=0, type=int, help='print frequency')
    parser.add_argument('--max-z', default=51, type=int, help='print frequency')
    parser.add_argument('--step', default=3, type=int, help='print frequency')
    parser.add_argument('--fg-dis', default=3, type=int, help='print frequency')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)