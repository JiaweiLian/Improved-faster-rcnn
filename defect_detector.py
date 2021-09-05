#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2021/6/19 15:58
# @Author  :Jiawei Lian 
# @FileName: defect_detector
# @Software: PyCharm

import os
import json
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import faster_rcnn
from faster_rcnn import FastRCNNPredictor, FasterRCNN
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from utils.engine import train_one_epoch, evaluate
from utils import transforms as T
from utils import utils
import time

"""
消除随机因素的影响
"""
torch.manual_seed(2021)

# model_path = './model_6.pth'
model_path = '/home/user/Public/Jiawei_Lian/HW_defect_detection/ckpt/0.5421/model_5_multi_scale_data.pth'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # parser.add_argument('--data_url', type=str, required=True,
    #                     help='the dataset dir of dataset')
    # parser.add_argument('--train_url', type=str, required=True,
    #                     help='the checkpoint dir obs')
    parser.add_argument('--init_method', type=str, required=False,
                        help='')
    parser.add_argument('--num_gpus', type=int, required=False, default=0,
                        help='')
    parser.add_argument('--last_path', type=str, required=False,
                        help='')
    parser.add_argument('--train-dir', type=str, required=False,
                        default='./data_mixed',
                        help='the dataset dir of training dataset')
    parser.add_argument('--validate-dir', type=str, required=False,
                        default='./data_mixed',
                        help='the dataset dir of validation dataset')
    parser.add_argument('--ckpt-dir', type=str, required=False,
                        default='./ckpt',
                        help='the checkpoint dir')

    parser.add_argument('--num-classes', type=int, required=False, default=10,
                        help='num-classes, do not include bg')
    parser.add_argument('--batch-size', type=int, required=False, default=4,
                        help='batch size')
    parser.add_argument('--num-epochs', type=int, required=False, default=20,
                        help='the number of epochs')
    parser.add_argument('--val-ratio', type=float, required=False, default=0.,
                        help='ratio of val set')
    argus = parser.parse_args()
    return argus


args = parse_args()


class CustomDataset(object):
    def __init__(self, root, transforms, ignore_area=20):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.ignore_area = ignore_area
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.label_mapping = {
            "connection_edge_defect": 1,
            "right_angle_edge_defect": 2,
            "cavity_defect": 3,
            "burr_defect": 4,
            "huahen": 5,
            "mosun": 6,
            "yanse": 7,
            'basi': 8,
            'jianju': 9,
            'chuizhidu': 10
        }

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "Annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB, because each color corresponds to a different instance, with 0 being background
        with open(annotation_path, 'r') as f:
            annotation_obj = json.load(f)
        # convert the PIL Image into a numpy array
        boxes = []
        labels = []
        area = []
        is_crowd = []
        for instance_obj in annotation_obj['shapes']:

            labels.append(self.label_mapping[instance_obj['label']])
            points = np.asarray(instance_obj['points'])
            cur_box = [
                np.min(points[:, 0]),
                np.min(points[:, 1]),
                np.max(points[:, 0]),
                np.max(points[:, 1])
            ]
            cur_area = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
            if cur_area < self.ignore_area:
                print('ignore small bbox < ' '{} {}'.format(self.ignore_area, os.path.basename(img_path)))
                continue
            boxes.append(cur_box)
            area.append(cur_area)
            is_crowd.append(0)
        boxes = np.asarray(boxes, np.float32)
        labels = np.asarray(labels, np.float32)
        is_crowd = np.asarray(is_crowd, np.int32)
        area = np.asarray(area, np.float32)
        target = dict()
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape([-1, 4])
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(is_crowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_train_transform():
    transforms = [
        T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.randomScale(),
        T.ToTensor(),
    ]
    return T.Compose(transforms)


def get_test_transform():
    transforms = [
        T.ToTensor(),
    ]
    return T.Compose(transforms)


def get_object_detector(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 1 + args.num_classes
    # use our dataset and defined transformations
    dataset = CustomDataset(args.train_dir, get_train_transform())

    if args.validate_dir is not None:
        dataset_test = CustomDataset(args.validate_dir, get_test_transform())
    else:
        dataset_test = None

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    num_train = int(len(dataset) * (1 - args.val_ratio))
    ##############################################################
    dataset = torch.utils.data.Subset(dataset, indices[:num_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])
    ##############################################################
    # define training and validation data loaders
    batch_size = args.batch_size
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    else:
        data_loader_test = None

    #############################################
    # data_loader_test = None
    #############################################

    # get the model using our helper function
    model = get_object_detector(num_classes)
    ######################################
    model.load_state_dict(
        torch.load(model_path, map_location='cuda')[
            'model'])
    ######################################

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=5e-4)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = args.num_epochs
    # local_ckpt_path = None
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if epoch > 20:
            if data_loader_test is not None:
                evaluate(model, data_loader_test, device=device)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        if args.ckpt_dir is not None:
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
                print('mkdir: {}'.format(args.ckpt_dir))
        local_ckpt_path = os.path.join(args.ckpt_dir, 'model_{}.pth'.format(epoch))
        utils.save_on_master(checkpoint, local_ckpt_path)


if __name__ == '__main__':

    start_time = time.time()
    main()
    end_time = time.time()
    print('Running time: %s minutes' % ((end_time - start_time) / 60))
