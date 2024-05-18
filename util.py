# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
# print(__dir__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, 'PP-HumanSeg/')))
# print(os.path.abspath(os.path.join(__dir__, './PP-HumanSeg/')))
from paddleseg.utils import get_sys_env, logger
from paddleseg.pre.infer import Predictor


import qrcode

import requests
token = "Bshb6dL7AE5OB7Xd7BD6IIdmwPrJcKTQ"



def get_bg_img(bg_img_path, img_shape):
    if bg_img_path is None:
        bg = 255 * np.ones(img_shape)
    elif not os.path.exists(bg_img_path):
        raise Exception('The --bg_img_path is not existed: {}'.format(
            bg_img_path))
    else:
        bg = cv2.imread(bg_img_path)
    return bg


def makedirs(save_dir):
    dirname = save_dir if os.path.isdir(save_dir) else \
        os.path.dirname(save_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def seg_camera(config, vertical_screen, bg_img_path):
    # 获取视频
    cap_camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    assert cap_camera.isOpened(), "Fail to open camera"
    width = int(cap_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(width,height) #640 480
    # 这里是换背景的地方
    bg = get_bg_img(bg_img_path, [height, width, 3])

    logger.info("Input: camera")
    logger.info("Create predictor...")
    predictor = Predictor(config, vertical_screen)

    logger.info("Start predicting...")
    while cap_camera.isOpened():
        ret_img, img = cap_camera.read()
        if not ret_img:
            break
        out = predictor.run(img, bg)
        cv2.imshow('PP-HumanSeg', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_camera.release()


def upload(path,qrpath):
    token = ''
    files = {'file': open(path, 'rb')}
    url = f'https://img.cloudtile.net/api/v1/upload'
    # 构建请求头
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
    }
    try:
        response = requests.post(url, headers=headers,files=files)
        response.raise_for_status()  # 检查是否有HTTP错误
        # 处理成功的情况，比如打印响应内容
        print("Response:", response.json())
        # 生成二维码
        download_url = response.json()['data']['links']['url']
        print(download_url)

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(download_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save(qrpath)

    except requests.exceptions.RequestException as e:
        # 处理请求异常，比如网络问题、超时等
        print(f"Request failed: {e}")

    except requests.exceptions.HTTPError as e:
        # 处理HTTP错误
        print(f"HTTP error: {e}")

    except requests.exceptions.Timeout as e:
        # 处理超时异常
        print(f"Request timed out: {e}")

    except requests.exceptions.ConnectionError as e:
        # 处理连接错误
        print(f"Connection error: {e}")



def show_real_button(image3_path):
    image3 = Image.open(image3_path)
    desired_width = 1080
    desired_height = 420
    # 图片3的裁剪或填充
    if image3.size[0] > desired_width or image3.size[1] > desired_height:
        # 裁剪图片3
        left = (image3.size[0] - desired_width) // 2
        top = (image3.size[1] - desired_height) // 2
        right = left + desired_width
        bottom = top + desired_height
        image3 = image3.crop((left, top, right, bottom))
    elif image3.size[0] == desired_width and image3.size[1]==420:
        image3 = image3
    else:
        # 填充图片3
        background = Image.new('RGB', (desired_width, desired_height), (255, 255, 255))
        background.paste(image3,
                         (int((desired_width - image3.size[0]) / 2), int((desired_height - image3.size[1]) / 2)))
        image3 = background

def show_real(text_file, image2, image3_path):
    # 创建图片1的中文文本
    with open(text_file, "r", encoding="utf-8") as file:
        chinese_texts = [line.strip() for line in file]
        chinese_text = chinese_texts[0].split(",")[1]
        font_path = chinese_texts[1].split(",")[1]
        font_size = chinese_texts[2].split(",")[1]
        font = chinese_texts[3].split(",")[1]
        font_size = chinese_texts[4].split(",")[1]

    info = [chinese_text,font,font_size]
    # 定义背景颜色的十六进制代码
    hex_color = "#000000"

    # 将十六进制颜色代码转换为RGB元组
    rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    # 创建图片1，白色背景
    width = 1080
    height = 420
    image1 = Image.new("RGB", (width, height),rgb_color )

    # 图片2的加载
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    # 创建最终的目标图片
    final_image = Image.new("RGB", (1080, 1920))

    # 将图片1、图片2、图片3依次粘贴到最终图片上
    final_image.paste(image1, (0, 0))
    final_image.paste(image2, (0, 420))
    # 加载图片3并调整大小
    image3 = Image.open(image3_path)
    desired_width = 1080
    desired_height = 420
    # 图片3的裁剪或填充
    if image3.size[0] > desired_width or image3.size[1] > desired_height:
        # 裁剪图片3
        left = (image3.size[0] - desired_width) // 2
        top = (image3.size[1] - desired_height) // 2
        right = left + desired_width
        bottom = top + desired_height
        image3 = image3.crop((left, top, right, bottom))
    elif image3.size[0] == desired_width and image3.size[1]==420:
        image3 = image3
    else:
        # 填充图片3
        background = Image.new('RGB', (desired_width, desired_height), (255, 255, 255))
        background.paste(image3,
                         (int((desired_width - image3.size[0]) / 2), int((desired_height - image3.size[1]) / 2)))
        image3 = background

    final_image.paste(image3, (0, 1500))
    fl = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
    # cv2.imwrite(self.real_pic_path)
    # del final_image
    return fl,info

def create_final_image(list_to_image, image2_path, image3_path,final_path,text_fill,bg_color="#000000"):
    # 创建图片1的中文文本
    chinese_text = list_to_image[0]
    font_path = list_to_image[1]
    font_size = list_to_image[2]
    hex_color = bg_color

    # 将十六进制颜色代码转换为RGB元组
    rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    # 创建图片1，白色背景
    width = 1080
    height = 420
    image1 = Image.new("RGB", (width, height), rgb_color)
    draw = ImageDraw.Draw(image1)

    fnt = ImageFont.truetype(font_path, int(font_size))
    text_width, text_height = draw.textsize(chinese_text, font=fnt)
    # 计算文本的位置
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    # 在图片1上绘制中文文本

    if text_fill is None:
        text_fill = 'black'
    draw.text((text_x, text_y), chinese_text, fill=text_fill, font=fnt)

    # 图片2的加载
    image2 = Image.open(image2_path)

    # 创建最终的目标图片
    final_image = Image.new("RGB", (1080, 1920))

    # 将图片1、图片2、图片3依次粘贴到最终图片上
    final_image.paste(image1, (0, 0))
    final_image.paste(image2, (0, 420))
    # 加载图片3并调整大小
    image3 = Image.open(image3_path)
    desired_width = 1080
    desired_height = 420
    # 图片3的裁剪或填充
    if image3.size[0] > desired_width or image3.size[1] > desired_height:
        # 裁剪图片3
        left = (image3.size[0] - desired_width) // 2
        top = (image3.size[1] - desired_height) // 2
        right = left + desired_width
        bottom = top + desired_height
        image3 = image3.crop((left, top, right, bottom))
    elif image3.size[0] == desired_width and image3.size[1]==420:
        image3 = image3
    else:
        # 填充图片3
        background = Image.new('RGB', (desired_width, desired_height), (255, 255, 255))
        background.paste(image3,
                         (int((desired_width - image3.size[0]) / 2), int((desired_height - image3.size[1]) / 2)))
        image3 = background

    final_image.paste(image3, (0, 1500))
    fl = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(final_path,fl)
    del final_image
    return fl



if __name__ == "__main__":
    # args = parse_args()
    # env_info = get_sys_env()
    # args.use_gpu = True if env_info['Paddle compiled with cuda'] \
    #     and env_info['GPUs used'] else False

    # makedirs(args.save_dir)
    config = r'E:\ALLCODE\Pythoncode\virtual_backgroud\simple2\inference_models\v2-lite-192\deploy.yaml'
    vertical_screen = False
    bg_img_path = r'E:\ALLCODE\Pythoncode\virtual_backgroud\simple2\data\5.jpg'
    # save_dir =
    seg_camera(config,vertical_screen,bg_img_path)

