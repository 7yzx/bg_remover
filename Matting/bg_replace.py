import argparse
import os
import sys

import cv2
import numpy as np
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(LOCAL_PATH, os.pardir))
PARENT_PATH2 = os.path.abspath(os.path.join(LOCAL_PATH, os.pardir,os.pardir))

# sys.path.append(PARENT_PATH)
# sys.path.append(PARENT_PATH2)

# print(sys.path)
import paddle
import paddleseg
from paddleseg.cvlibs import manager
manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

# import ppmatting
# from ppmatting.core import predict
# from ppmatting.utils import get_image_list, estimate_foreground_ml, Config, MatBuilder


# def mattingprocess(getcfg, device, model_path, image_path, save_dir, background):
#     cfg = Config(getcfg)
#     builder = MatBuilder(cfg)
#
#     # paddleseg.utils.show_env_info()
#     paddleseg.utils.show_cfg_info(cfg)
#     # paddleseg.utils.set_device(device)
#
#     model = builder.model
#     transforms = ppmatting.transforms.Compose(builder.val_transforms)
#
#     alpha, fg = predict(
#         model,
#         model_path=model_path,
#         transforms=transforms,
#         image_list=[image_path],
#         trimap_list=[None],
#         save_dir=save_dir,
#         fg_estimate=True)
#
#     img_ori = cv2.imread(image_path)
#     bg = get_bg(background, img_ori.shape)
#     alpha = alpha / 255.0
#     alpha = alpha[:, :, np.newaxis]
#     com = alpha * fg + (1 - alpha) * bg
#     com = com.astype('uint8')
#     com_save_path = os.path.join(save_dir,
#                                  os.path.basename(image_path))
#     cv2.imwrite(com_save_path, com)
#     return com_save_path

def get_bg(background):
    if not os.path.exists(background):
        bg = np.zeros((1920, 1080, 3))
        bg[:, :, 2] = 255
    else:
        bg = cv2.imread(background)
    return bg

if __name__ == "__main__":
    background ='demo/5.jpg '
    getcfg = 'configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml'
    image_path = 'demo/human.jpg'
    model_path = 'pretrained_models/ppmattingv2-stdc1-human_512.pdparams'
    save_dir = '../output'
    device = 'cpu'
    # args = parse_args()
    # mattingprocess(getcfg,device,model_path,image_path,save_dir,background)
