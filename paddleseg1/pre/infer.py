# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import os
import time

import yaml
import numpy as np
import cv2
import paddle
from paddle.inference import create_predictor
from paddle.inference import Config as PredictConfig


import paddleseg1.transforms as T
from paddleseg1.infer import reverse_transform
from paddleseg1 import manager
from paddleseg1.timer import TimeAverager

from paddleseg1.pre.optic_flow_process import optic_flow_process

from paddleseg1.utils import logger

class DeployConfig:
    def __init__(self, path, vertical_screen):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

            [width, height] = self.dic['Deploy']['transforms'][0]['target_size']
            if vertical_screen and width > height:
                self.dic['Deploy']['transforms'][0][
                    'target_size'] = [height, width]

        self._transforms = self._load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def target_size(self):
        [width, height] = self.dic['Deploy']['transforms'][0]['target_size']
        return [width, height]

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return transforms


class Predictor:
    def __init__(self, config,vertical_screen):
        self.cfg = DeployConfig(config, vertical_screen)
        self.compose = T.Compose(self.cfg.transforms)
        gpu_available = paddle.device.is_compiled_with_cuda()

        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg.disable_glog_info()
        if gpu_available:
            pred_cfg.enable_use_gpu(2000, 0)
            print("----------------------------use gpu")
        self.predictor = create_predictor(pred_cfg)
        self.disflow = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        width, height = self.cfg.target_size()
        self.prev_gray = np.zeros((height, width), np.uint8)
        self.prev_cfd = np.zeros((height, width), np.float32)
        self.is_first_frame = True


    def run(self, img, bg,fac_x,fac_y,pos_x,pos_y):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])

        data = self.compose({'img': img})
        input_data = np.array([data['img']])

        input_handle.reshape(input_data.shape)
        input_handle.copy_from_cpu(input_data)

        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output = output_handle.copy_to_cpu()
        # cv2.imwrite("predict_img.jpg",img)
        return self.postprocess(output, img, data, bg,fac_x,fac_y,pos_x,pos_y)

    def resize_after_predict(self, alpha,origin_img, bg, fac_x, fac_y, pos_x, pos_y):
        '''
        fac_x:x方向缩放的大小设置
        fac_y:y方向缩放的大小设置
        posx：x方向移动
        posy：y方向移动
        alpha:就是裁剪之后的img
        '''
        bg_h, bg_w = bg.shape[0], bg.shape[1]
        # new_alpha_layer(1920,1080),范围0-1
        new_alpha_layer = np.zeros((bg_h,bg_w),dtype=np.float32)
        alpha_height, alpha_width,_ = alpha.shape
        # print("alpah:h,w",alpha_height,alpha_width)
        resize_a = cv2.resize(alpha, (int(alpha_width * fac_x), int(alpha_height * fac_y)))
        resize_h, resize_w = resize_a.shape

        y = max(0, min(pos_y, int(bg_h - resize_h-1)))
        start_y = int(bg_h - resize_h - y)
        end_y = start_y + resize_h

        x = min(np.abs(pos_x), int((bg_w - resize_w)/2) - 1)
        if pos_x>0:
            start_x = int((bg_w - resize_w) / 2 + x)
        else:
            start_x = int((bg_w - resize_w) / 2 - x)
        end_x = start_x + resize_w
        new_alpha_layer[start_y:end_y, start_x:end_x] = resize_a
        # （1920，1080，1）
        new_alpha_layer = new_alpha_layer[:, :, np.newaxis]

        new_image_layer = np.zeros( ( bg.shape ), dtype=np.uint8)
        new_image_height, new_image_width, _ = new_image_layer.shape
        resize_i = cv2.resize(origin_img, (int(alpha_width * fac_x), int(alpha_height * fac_y)))
        resize_ih, resize_iw, _ = resize_i.shape

        iy = max(0, min(pos_y, int(bg_h - resize_ih)))
        start_iy = int(bg_h - resize_ih - iy)
        end_iy = start_iy + resize_ih
        ix = min(np.abs(pos_x), int((bg_w - resize_iw)/2) - 1)
        if pos_x>0:
            start_ix = int((bg_w - resize_w) / 2 + ix)
        else:
            start_ix = int((bg_w - resize_w) / 2 - ix)
        end_ix = start_ix + resize_iw
        new_image_layer[start_iy:end_iy, start_ix:end_ix, :] = resize_i
        return new_alpha_layer, new_image_layer

    def postprocess(self, pred_img, origin_img, data, bg,fac_x,fac_y,pos_x,pos_y):
        trans_info = data['trans_info']
        score_map = pred_img[0, 1, :, :]

        # post process
        mask_original = score_map.copy()
        mask_original = (mask_original * 255).astype("uint8")
        _, mask_thr = cv2.threshold(mask_original, 200, 1,
                                    cv2.THRESH_BINARY)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
        mask_erode = cv2.erode(mask_thr, kernel_erode)
        mask_dilate = cv2.dilate(mask_erode, kernel_dilate)
        score_map *= mask_dilate

        score_map = 255 * score_map
        cur_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.resize(cur_gray,
                              (pred_img.shape[-1], pred_img.shape[-2]))
        optflow_map = optic_flow_process(cur_gray, score_map, self.prev_gray, self.prev_cfd, \
                self.disflow, self.is_first_frame)
        self.prev_gray = cur_gray.copy()
        self.prev_cfd = optflow_map.copy()
        self.is_first_frame = False
        score_map = optflow_map / 255.

        score_map = score_map[np.newaxis, np.newaxis, ...]
        score_map = reverse_transform(
            paddle.to_tensor(score_map), trans_info, mode='bilinear')
        # h，w，1
        alpha = np.transpose(score_map.numpy().squeeze(1), [1, 2, 0])
        # h，w 大小在0-1之间，需要映射到255的大小
        # new_alpha = np.squeeze(alpha*255).astype(np.uint8)
        # cv2.imwrite("predict_alpha.jpg", new_alpha)
        h, w, _ = origin_img.shape
        # bg_h, bg_w = bg.shape[0], bg.shape[1]
        # # new_alpha_layer(1920,1080),范围0-1
        # new_alpha_layer = np.zeros((bg_h,bg_w),dtype=np.float32)
        # alpha_height, alpha_width,_ = alpha.shape
        # fac_x = 0.8
        # fac_y = 0.8
        # resize_a = cv2.resize(alpha, (int(alpha_width * fac_x), int(alpha_height * fac_y)))
        # resize_h, resize_w = resize_a.shape
        # start_y = int(bg_h - resize_h)
        # end_y = start_y + resize_h
        # start_x = int((bg_w - resize_w) / 2)
        # end_x = start_x + resize_w
        # new_alpha_layer[start_y:end_y, start_x:end_x] = resize_a
        # # （1920，1080，1）
        # new_alpha_layer = new_alpha_layer[:, :, np.newaxis]
        #
        # new_image_layer = np.zeros( ( bg.shape ), dtype=np.uint8)
        # new_image_height, new_image_width, _ = new_image_layer.shape
        # resize_i = cv2.resize(origin_img, (int(alpha_width * fac_x), int(alpha_height * fac_y)))
        # resize_ih, resize_iw, _ = resize_i.shape
        # start_iy = int(bg_h - resize_ih)
        # end_iy = start_iy + resize_ih
        # start_ix = int((bg_w - resize_iw) / 2)
        # end_ix = start_ix + resize_iw
        # new_image_layer[start_iy:end_iy, start_ix:end_ix, :] = resize_i
        new_alpha_layer, new_image_layer = self.resize_after_predict(alpha, origin_img, bg, fac_x, fac_y, pos_x, pos_y)
        # （1920，1080，1）
        # bg = cv2.resize(bg, (w, h))

        if bg.ndim == 2:
            bg = bg[..., np.newaxis]
        # fg = (new_alpha_layer * new_image_layer).astype(np.uint8)
        # cv2.imwrite("predict_fg.jpg", fg)
        # out = (alpha * origin_img + (1 - alpha) * bg).astype(np.uint8)
        out = (new_alpha_layer * new_image_layer + (1 - new_alpha_layer) * bg).astype(np.uint8)
        # cv2.imwrite("predict_out.jpg", out)
        return out
