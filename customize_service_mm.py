# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import numpy as np
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from model_service.pytorch_model_service import PTServingBaseService

from tqdm import tqdm
from collections import OrderedDict
###### 需要修改Begin####
# if torch.cuda.is_available():
#     os.system(f'nvcc -V')
# else:
#     print('No cuda found!!!!!!')
# os.system(f'pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')
# config_file = 'configs/0712_vfnet_r50_fpn_mdconv_c3-c5_mstrain_14e_coco.py'
config_file = 'configs/0715_mini_cascade_rcnn_r50_fpn_soft_cutout_context.py'
# config_file = 'configs/0712_albu_vfnet_r50_fpn_mdconv_c3-c5_mstrain_14e_coco.py'
if torch.cuda.is_available():
    mmcv_path = os.path.join(os.path.dirname(__file__), 'mmcv_full-1.2.5+torch1.6.0+cu101-cp36-cp36m-manylinux1_x86_64.whl')
else:
    mmcv_path = os.path.join(os.path.dirname(__file__), 'mmcv_full-1.2.5+torch1.6.0+cpu-cp36-cp36m-manylinux1_x86_64.whl')

###### 需要修改End######
# mmdet_path = os.path.join(os.path.dirname(__file__), 'mmdet-2.10.0-py3-none-any.whl')
# try:
#     os.system(f'ls -l /usr/local | grep cuda')
# except:
#     print('BAD: ls -l /usr/local | grep cuda')
# try:
#     os.system(f'sudo ln -snf /usr/local/cuda-10.1 /usr/local/cuda')
# except:
#     print('BAD: sudo ln -snf /usr/local/cuda-10.1 /usr/local/cuda')
try:
    os.system(f'pip uninstall mmcv-full -y')
except:
    print('BAD: pip uninstall mmcv-full -y')
os.system(f'pip install {mmcv_path}')
# os.system(f'pip install {mmcv_path}; pip install {mmdet_path}')
# os.system(f'pip install {mmdet_path}')
from mmdet.apis import inference_detector, init_detector
import warnings
print('mmdet import success')

import time
from metric.metrics_manager import MetricsManager
from torchvision.transforms import functional as F
import log
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

logger.info(torch.__version__)
logger.info(torchvision.__version__)


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path, **kwargs):
        self.input_image_key = "input_img"
        self.score_thr = 0.
        self.model_name = model_name
        # self.checkpoint = os.path.join(os.path.dirname(__file__), 'model.pth')
        self.checkpoint = model_path
        self.config = os.path.join(os.path.dirname(__file__), config_file)
        # num_classes = 1 + 10
        logger.info('{}-{}'.format(self.checkpoint, self.config))
        for key in kwargs:
            logger.info('{}-{}'.format(key, kwargs[key]))
        # self.model_path = model_path
        self.use_cuda = True
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        logger.info('remove pretrained')
        if torch.cuda.is_available():
            logger.info('Using GPU for inference')
        else:
            logger.info('Using CPU for inference')
        self.model = init_detector(self.config, self.checkpoint, device=self.device)
        # self.model.eval()
        self.classes = self.model.CLASSES
        print("model already")

    def _preprocess(self, data):
        # # img = data
        # img = Image.open(data).convert("RGB")
        # img = np.array(img)
        # # return torch.unsqueeze(F.to_tensor(img), dim=0).to(device)
        # return img
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content).convert("RGB")
                img = np.array(img)
                preprocessed_data[k] = [img]
                # image = Image.open(file_content)
                # preprocessed_data[k] = [image]
        return preprocessed_data

    def _inference(self, data):
        img = data[self.input_image_key]
        # print(data) # wenht
        result_my = inference_detector(self.model, img)
        result = []
        for r in result_my:
            result.append(self.result2final(r))
        # for idx in range(len(result)):
        #     result[idx]['boxes'] = result[idx][
        #         'boxes'].cpu().detach().numpy().tolist()
        #     result[idx]['labels'] = result[idx][
        #         'labels'].cpu().detach().numpy().tolist()
        #     result[idx]['scores'] = result[idx][
        #         'scores'].cpu().detach().numpy().tolist()
        result = {"result": result}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
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

    def result2final(self, result_my):
        bboxes = np.vstack(result_my)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result_my)
        ]
        labels = np.concatenate(labels)
        inds = np.where(bboxes[:, -1] > self.score_thr)
        bboxes = bboxes[inds]
        labels = labels[inds]
        # result = OrderedDict()
        result = {}
        if len(bboxes) > 0:
            out_classes = labels
            out_scores = bboxes[:, 4]
            out_boxes = bboxes[:, 0:4]

            detection_class_names = []
            for class_id in out_classes:
                # class_name = classes[int(class_id)]
                class_name = int(class_id)+1
                detection_class_names.append(class_name)

            out_boxes_list = []
            for box in out_boxes:
                out_boxes_list.append([round(float(v), 1) for v in box])

            result['boxes'] = out_boxes_list
            result['labels'] = detection_class_names
            result['scores'] = [round(float(v), 4) for v in out_scores]
        else:
            result['boxes'] = []
            result['labels'] = []
            result['scores'] = []
        return result