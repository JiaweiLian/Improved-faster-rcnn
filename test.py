#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2021/6/19 15:58
# @Author  :Jiawei Lian
# @FileName: defect_detector
# @Software: PyCharm
from copy import deepcopy
import random
import cv2
import ensemble_boxes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.hub import load_state_dict_from_url
# from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F, transforms
from torchvision.transforms import transforms as T

import faster_rcnn


class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 512

    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes, image):
        raise NotImplementedError


def get_object_detector(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes, image):
        width = image.width
        boxes[:, [2, 0]] = width - boxes[:, [0, 2]]
        return boxes


class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image

    def batch_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes, image):
        height = image.height
        boxes[:, [3, 1]] = height - boxes[:, [1, 3]]
        return boxes


class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """

    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes
        boxes[:, 0], idx = result_boxes[:, [0, 2]].min(1)
        boxes[:, 0], idx = result_boxes[:, [0, 2]].min(1)
        boxes[:, 0], idx = result_boxes[:, [0, 2]].min(1)
        boxes[:, 0], idx = result_boxes[:, [0, 2]].min(1)
        return boxes

    def deaugment_boxes(self, boxes, image):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes, image)
        return self.prepare_boxes(boxes)


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def del_under_threshold(result, threshold=0.):
    idxes = []
    for idx in range(len(result[0]['scores'])):
        if result[0]['scores'][idx] < threshold:
            idxes.append(idx)
    for i in idxes:
        result[0]['scores'] = del_tensor_ele(result[0]['scores'], len(result[0]['scores']) - 1)
        result[0]['labels'] = del_tensor_ele(result[0]['labels'], len(result[0]['labels']) - 1)
        result[0]['boxes'] = del_tensor_ele(result[0]['boxes'], len(result[0]['boxes']) - 1)
    return result


def del_fusion_under_threshold(boxes, labels, scores, threshold=0.):
    idxes = []
    for idx in range(len(scores)):
        if scores[idx] < threshold:
            idxes.append(idx)
    for i in idxes:
        scores = del_tensor_ele(scores, len(scores) - 1)
        labels = del_tensor_ele(labels, len(labels) - 1)
        boxes = del_tensor_ele(boxes, len(boxes) - 1)
    return boxes, labels, scores


def py_cpu_nms(boxes, scores, thresh=0.1):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    boxes = boxes.detach().numpy()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = scores

    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score置信度降序排序
    # order = scores.argsort()[::-1]
    all_scores, order = scores.sort(descending=True)

    keep = []  # 保留的结果框集合
    # print(order)
    while int(len(order.detach().numpy())) > 0:
        i = order[0]
        keep.append(i.numpy())  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    return keep


def soft_nms(bboxes, scores, Nt=0.3, sigma2=0.5, score_thresh=0.001, method=2):
    # 在 bboxes 之后添加对于的下标[0, 1, 2...], 最终 bboxes 的 shape 为 [n, 5], 前四个为坐标, 后一个为下标
    # res_bboxes = deepcopy(bboxes)
    N = bboxes.shape[0]  # 总的 box 的数量
    indexes = np.array([np.arange(N)])  # 下标: 0, 1, 2, ..., n-1
    bboxes = bboxes.detach().numpy()
    bboxes = np.concatenate((bboxes, indexes.T), axis=1)  # concatenate 之后, bboxes 的操作不会对外部变量产生影响
    # 计算每个 box 的面积

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores, order = scores.sort(descending=True)
    scores = scores.detach().numpy()

    for i in range(N):
        # 找出 i 后面的最大 score 及其下标
        pos = i + 1
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        # 如果当前 i 的得分小于后面的最大 score, 则与之交换, 确保 i 上的 score 最大
        if scores[i] < maxscore:
            bboxes[[i, maxpos + i + 1]] = bboxes[[maxpos + i + 1, i]]
            scores[[i, maxpos + i + 1]] = scores[[maxpos + i + 1, i]]
            areas[[i, maxpos + i + 1]] = areas[[maxpos + i + 1, i]]
        # IoU calculate
        xx1 = np.maximum(bboxes[i, 0], bboxes[pos:, 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[pos:, 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[pos:, 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[pos:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[pos:] - intersection)
        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(iou.shape)
            weight[iou > Nt] = weight[iou > Nt] - iou[iou > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(iou * iou) / sigma2)
        else:  # original NMS
            weight = np.ones(iou.shape)
            weight[iou > Nt] = 0
        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    inds = bboxes[:, 4][scores > score_thresh]
    keep = inds.astype(int)
    return keep


# image_path = './data/Images/2020-01-11_21_43_14_145.jpg'
# image_path = './data/Images/2020-03-07_08_34_30_467.jpg'
# image_path = './data/Images/2020-01-11_21_41_15_002.jpg'
# image_path = './data/Images/2020-01-11_21_36_02_642.jpg'
# image_path = './data/Images/2020-03-10_16_18_20_688.jpg'
image_path = './data/Images/2021-05-29-18-44-02.jpg'
# image_path = './data/Images/2021-05-16-18-51-54.jpg'
# image_path = './data/Images/2021-05-16-14-58-28.jpg'

model_path = '/home/user/Public/Jiawei_Lian/HW_defect_detection/ckpt/0.5959/model_23_5959_5288.pth'
# model_path = '/home/user/Public/Jiawei_Lian/HW_defect_detection/ckpt/model_0.pth'

results = []
predictions = []

# you can try own combinations:
transform1 = TTACompose([
    TTAHorizontalFlip(),
    # TTAVerticalFlip()

])

transform2 = TTACompose([
    # TTAHorizontalFlip(),
    TTAVerticalFlip()
])

fig, ax = plt.subplots(3, 2, figsize=(16, 10))

image1 = Image.open(image_path).convert("RGB")
image1_vf = F.vflip(image1)

image_tensor = torch.from_numpy(np.array(image1))
image_tensor_vf = torch.from_numpy(np.array(image1_vf))
# image_tensor = image_tensor.permute(0, 1, 2)
image_numpy_vf = image_tensor_vf.cpu().numpy().copy()
image_numpy = image_tensor.cpu().numpy().copy()
image_numpy1 = image_tensor.cpu().numpy().copy()
image_numpy2 = image_tensor.cpu().numpy().copy()
image_numpy3 = image_tensor.cpu().numpy().copy()

# ax[0, 0].imshow(image)
# ax[0, 0].set_title('original')

tta_image1 = transform1.augment(image_tensor)
tta_image2 = transform2.augment(image_tensor_vf)

tta_image1_numpy = tta_image1.numpy().copy()
tta_image2_numpy = image_tensor_vf.numpy().copy()

tta_image1 = Image.fromarray(tta_image1_numpy)
tta_image2 = Image.fromarray(tta_image2_numpy)

########################################################################
# tta_image1 prediction
preprocessed_image = torch.unsqueeze(F.to_tensor(tta_image1), dim=0)
model = get_object_detector(11)
model.load_state_dict(
    torch.load(model_path, map_location='cpu')[
        'model'])
model.eval()
result = model(preprocessed_image)
result = del_under_threshold(result)
print('tta_image prediction:', result)

boxes3 = result[0]['boxes']
scores3 = result[0]['scores']
labels3 = result[0]['labels']

for box in boxes3:
    cv2.rectangle(tta_image1_numpy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

ax[0, 0].imshow(tta_image1_numpy)
ax[0, 0].set_title('Augment1')

###################################################################
# deaugmentation prediction
boxes3 = transform1.deaugment_boxes(boxes3, image1)
results.append({
    'boxes': boxes3,
    'scores': scores3,
    'labels': labels3,
})

for box in boxes3:
    cv2.rectangle(image_numpy1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

ax[0, 1].imshow(image_numpy1)
ax[0, 1].set_title('Deaugment1')
#########################################################

########################################################################
# tta_image2 prediction
preprocessed_image = torch.unsqueeze(F.to_tensor(tta_image2), dim=0)
model = get_object_detector(11)
model.load_state_dict(
    torch.load(model_path, map_location='cpu')[
        'model'])
model.eval()
result = model(preprocessed_image)
result = del_under_threshold(result)
print('tta_image prediction:', result)

boxes4 = result[0]['boxes']
scores4 = result[0]['scores']
labels4 = result[0]['labels']

for box in boxes4:
    cv2.rectangle(tta_image2_numpy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

ax[1, 0].imshow(tta_image2_numpy)
ax[1, 0].set_title('Augment2')

###################################################################
# deaugmentation prediction
boxes4 = transform2.deaugment_boxes(boxes4, image1_vf)
results.append({
    'boxes': boxes4,
    'scores': scores4,
    'labels': labels4,
})

for box in boxes4:
    cv2.rectangle(image_numpy3, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

ax[1, 1].imshow(image_numpy3)
ax[1, 1].set_title('Deaugment2')

#########################################################
# original_image prediction
############################################################
# random scale
scale_width = random.uniform(0.85, 1.15)
scale_height = random.uniform(0.85, 1.15)
width = image1.width
height = image1.height
image1 = np.array(image1)
image1 = cv2.resize(image1, (int(width * scale_width), int(height * scale_height)))
image_scaled_tensor = torch.from_numpy(np.array(image1))
image_scaled_numpy = image_scaled_tensor.cpu().numpy().copy()
############################################################
preprocessed_image = torch.unsqueeze(F.to_tensor(image1), dim=0)
model = get_object_detector(11)
model.load_state_dict(
    torch.load(model_path, map_location='cpu')[
        'model'])
model.eval()
result_original_image = model(preprocessed_image)
result_original_image = del_under_threshold(result_original_image)

print('original image prediction:', result_original_image)

boxes2 = result_original_image[0]['boxes']
scores2 = result_original_image[0]['scores']
labels2 = result_original_image[0]['labels']
results.append({
    'boxes': boxes2,
    'scores': scores2,
    'labels': labels2,
})

scale_tensor = torch.FloatTensor([[1.0 / scale_width, 1.0 / scale_height, 1.0 / scale_width, 1.0 / scale_height]]).expand_as(boxes2)
boxes2 = boxes2 * scale_tensor

for box in boxes2:
    cv2.rectangle(image_numpy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

ax[2, 0].imshow(image_numpy)
ax[2, 0].set_title('Original')

#######################################################
# # weghted boxes fusion
temp_all_boxes = torch.cat((boxes3, boxes2, boxes4), 0)
all_labels = torch.cat((labels3, labels2, labels4), 0)
all_scores = torch.cat((scores3, scores2, scores4), 0)

_, indices = all_scores.sort(descending=True)
all_labels = all_labels.gather(dim=0, index=indices)
all_scores = all_scores.gather(dim=0, index=indices)
all_boxes = torch.empty(len(indices), 4)
for i in range(len(indices)):
    all_boxes[i] = temp_all_boxes[indices[i]]

all_boxes, all_labels, all_scores = del_fusion_under_threshold(all_boxes, all_labels, all_scores)

keep = py_cpu_nms(all_boxes, all_scores)
# keep = soft_nms(all_boxes, all_scores)

all_scores1 = all_scores[:len(keep)]
all_labels1 = all_labels[:len(keep)]
all_boxes1 = all_boxes[:len(keep)]

for i in range(len(keep)):
    all_scores1[i] = all_scores[keep[i]]
    all_labels1[i] = all_labels[keep[i]]
    all_boxes1[i] = all_boxes[keep[i]]

labels = ["",
          "connection_edge_defect",
          "right_angle_edge_defect",
          "cavity_defect",
          "burr_defect",
          "huahen",
          "mosun",
          "yanse",
          'basi',
          'jianju',
          'chuizhidu', ]
i = 0
for box in all_boxes1:
    cv2.rectangle(image_numpy2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

ax[2, 1].imshow(image_numpy2)
ax[2, 1].set_title('Fusion')

# Image._show(Image.fromarray(image_numpy2))
# Image.fromarray(image_numpy2).save('prediction.jpg')

print('fusion prediction:')
print(all_labels1)
print(all_scores1)
print(all_boxes1)
