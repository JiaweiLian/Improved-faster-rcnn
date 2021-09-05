# -*- coding: utf-8 -*-
import os
import random

import cv2
import torch
import torchvision

import faster_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model_service.pytorch_model_service import PTServingBaseService
import ensemble_boxes
import numpy as np
import torch
import time
from metric.metrics_manager import MetricsManager
from torchvision.transforms import functional as F, transforms
import log
import json
from PIL import Image
from torchvision.transforms import transforms as T


Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

logger.info(torch.__version__)
logger.info(torchvision.__version__)


def get_object_detector(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    logger.info('remove pretrained')
    model = faster_rcnn.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    logger.info('{}-{}'.format(in_features, num_classes))
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


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


def py_cpu_nms(boxes, scores, thresh):
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


class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 512

    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes, image):
        raise NotImplementedError


class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes, image):
        width = image.width
        height = image.height
        boxes[:, [2, 0]] = width - boxes[:, [0, 2]]
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


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path, **kwargs):
        self.model_name = model_name
        num_classes = 1 + 10
        logger.info('{}-{}'.format(num_classes, model_path))
        for key in kwargs:
            logger.info('{}-{}'.format(key, kwargs[key]))
        self.model_path = model_path
        self.model = get_object_detector(num_classes)
        self.use_cuda = False
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logger.info('Using GPU for inference')
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.model_path)['model'])
        else:
            logger.info('Using CPU for inference')
            self.model.load_state_dict(torch.load(self.model_path,
                                                  map_location='cpu')['model'])
        self.model.eval()
        print("model already")

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content).convert("RGB")
                preprocessed_data[k] = torch.unsqueeze(F.to_tensor(img),
                                                       dim=0).to(self.device)
        return preprocessed_data

    def _inference(self, data):

        img = data["input_img"]

        # torch.Tensor
        preprocessed_image = img

        # Original image
        image = tensor_to_PIL(preprocessed_image)  # PIL.Image format
        image_tensor = torch.from_numpy(np.array(image))
        image_tensor = image_tensor.permute(0, 1, 2)

        transform = TTACompose([
            TTAHorizontalFlip(),
        ])
        # tta_image
        tta_image = transform.augment(image_tensor)
        # tta_image = tta_image.permute(1, 2, 0)
        tta_image_numpy = tta_image.numpy().copy()
        tta_image1 = Image.fromarray(tta_image_numpy)

        # original image prediction
        result = self.model(preprocessed_image)

        boxes3 = result[0]['boxes']
        scores3 = result[0]['scores']
        labels3 = result[0]['labels']

        # augmented
        preprocessed_image = torch.unsqueeze(F.to_tensor(tta_image1), dim=0)
        result = self.model(preprocessed_image)

        boxes2 = result[0]['boxes']
        boxes2 = transform.deaugment_boxes(boxes2, image)
        scores2 = result[0]['scores']
        labels2 = result[0]['labels']

        # random scale
        scale_width = random.uniform(0.97, 1.03)
        scale_height = random.uniform(0.97, 1.03)
        width = image.width
        height = image.height
        image1 = np.array(image)
        image1 = cv2.resize(image1, (int(width * scale_width), int(height * scale_height)))
        preprocessed_image = torch.unsqueeze(F.to_tensor(image1), dim=0)
        result = self.model(preprocessed_image)

        boxes4 = result[0]['boxes']
        scale_tensor = torch.FloatTensor(
            [[1.0 / scale_width, 1.0 / scale_height, 1.0 / scale_width, 1.0 / scale_height]]).expand_as(boxes4)
        boxes4 = boxes4 * scale_tensor
        scores4 = result[0]['scores']
        labels4 = result[0]['labels']

        # fusion
        temp_all_boxes = torch.cat((boxes3, boxes2, boxes4), 0)
        # all_boxes = torch.cat((boxes3, boxes2), 0)
        all_labels = torch.cat((labels3, labels2, labels4), 0)
        all_scores = torch.cat((scores3, scores2, scores4), 0)

        _, indices = all_scores.sort(descending=True)
        all_labels = all_labels.gather(dim=0, index=indices)
        all_scores = all_scores.gather(dim=0, index=indices)
        all_boxes = torch.empty(len(indices), 4)
        for i in range(len(indices)):
            all_boxes[i] = temp_all_boxes[indices[i]]

        keep = py_cpu_nms(all_boxes, all_scores, thresh=0.55)
        # keep = soft_nms(all_boxes, all_scores)

        all_scores1 = all_scores[:len(keep)]
        all_labels1 = all_labels[:len(keep)]
        all_boxes1 = all_boxes[:len(keep)]

        for i in range(len(keep)):
            all_scores1[i] = all_scores[keep[i]]
            all_labels1[i] = all_labels[keep[i]]
            all_boxes1[i] = all_boxes[keep[i]]

        for idx in range(len(result)):
            result[idx]['boxes'] = all_boxes1.tolist()
            result[idx]['labels'] = all_labels1.tolist()
            result[idx]['scores'] = all_scores1.tolist()
        result = {"result": result}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data
