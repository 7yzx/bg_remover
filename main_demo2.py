# -*- coding: utf-8 -*-
import util
from demo2 import Ui_mainWindow
from PIL import Image
from PyQt5.QtCore import Qt, QTimer,pyqtSignal,QThread
from PyQt5.QtGui import QImage, QPixmap,QFont
from PyQt5.QtWidgets import QApplication,QVBoxLayout, QWidget,QMainWindow,QFileDialog,QMenu,QButtonGroup,QLabel
import cv2
import sys, os
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import autopy
import json
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(LOCAL_PATH, "Matting"))
sys.path.append(PARENT_PATH)
from paddleseg1.pre.infer import Predictor
from Matting.bg_replace import get_bg
import Matting.ppmatting as ppmatting
from Matting.ppmatting.core import predict
from Matting.ppmatting.utils import Config, MatBuilder

class ShowqrWindow(QWidget):
    closed = pyqtSignal(bool)  # 自定义信号，用于通知UI是否关闭

    def __init__(self, image_path):
        super(ShowqrWindow, self).__init__()
        self.is_close = False

        self.setWindowTitle("下载二维码")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)  # 仅显示关闭按钮
        self.setGeometry(0,0, 600, 500)
        self.setWindowOpacity(0.8)

        # Move the window to the center of the screen
        screen_geometry = QApplication.desktop().screenGeometry()
        # x = (screen_geometry.width() - self.width()) // 2
        # y = (screen_geometry.height() - self.height()) // 2
        x = (screen_geometry.width() - self.width()-50)
        y = (screen_geometry.height()-self.height()-100)
        self.move(x, y)

        layout = QVBoxLayout()

        # QLabel for displaying the image
        self.image_label = QLabel()
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)  # 图片居中显示
        layout.addWidget(self.image_label)

        # QLabel for displaying the countdown
        self.countdown_label = QLabel("请用微信扫码下载\n30 秒后自动关闭")
        layout.addWidget(self.countdown_label)
        font = QFont()
        font.setPointSize(40)
        self.countdown_label.setFont(font)
        self.countdown_label.setStyleSheet("color: black;")  # 设置字体颜色为红色
        self.countdown_label.setAlignment(Qt.AlignCenter)  # 将文本居中显示

        self.setLayout(layout)

        # Timer for countdown
        self.timerq = QTimer(self)
        self.timerq.timeout.connect(self.update_countdown)
        self.timerq.start(1000)  # Timer fires every 1000 milliseconds (1 second)
        self.countdown_seconds = 30

    def update_countdown(self):
        self.countdown_seconds -= 1
        self.countdown_label.setText(f"请用微信扫码下载\n{self.countdown_seconds} 秒后自动关闭")

        if self.countdown_seconds == 0:
            self.timerq.stop()
            self.is_close = True  # 更新关闭状态
            self.closed.emit(self.is_close)  # 发送信号，通知UI已关闭
            self.close()

    def closeEvent(self, event):
        self.is_close = True
        self.closed.emit(self.is_close)  # 发送信号，通知UI已关闭
        event.accept()


def load_json_with_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取 JSON 数据
        data = json.load(f)
        # 过滤掉注释键
        data = {k: v for k, v in data.items() if not k.startswith('_comment')}
        return data


def save_json_after(file_path,data):
    # 将修改后的数据写入 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class Image_predict_Thread(QThread):
    # 定义信号，用于传递处理结果
    predict_finished = pyqtSignal(object)

    def __init__(self, img, bg, src_factorx, src_factory, position_x, position_y, predictor):
        super(Image_predict_Thread, self).__init__()
        self.img = img
        self.bg = bg
        self.src_factorx = src_factorx
        self.src_factory = src_factory
        self.position_x = position_x
        self.position_y = position_y
        self.predictor = predictor

    def run(self):
        # 执行耗时的图像处理操作
        processed_img = self.process_image(self.img)
        # 发射信号，传递处理结果
        self.predict_finished.emit(processed_img)

    def process_image(self, img):
        # 在这里执行图像处理操作，例如图像抠图、背景替换等
        out = self.predictor.run(img, self.bg, self.src_factorx, self.src_factory, self.position_x, self.position_y)
        return out



class CameraApp(QMainWindow):
    def __init__(self):
        super(CameraApp, self).__init__()
        base_path = os.path.abspath(".") # 当前工作目录的路径
        self.bash_path = base_path
        # print(base_path)# 根目录
        # 读取json文件
        self.json_path = os.path.join(self.bash_path,"configs/local_config.json")
        self.local_configs = load_json_with_comments(self.json_path)
        self.text_path = os.path.join(self.bash_path,"text.json")
        self.config_text = load_json_with_comments(self.text_path)
        self.welcome_words_in_realtime_show = self.config_text["welcome_words_in_realtime_show"]
        self.welcome_words_in_pic_show = self.config_text["welcome_words_in_pic_show"]

        self.real_time_font_size = self.config_text["real_time_font_size"]
        self.real_time_font_path = self.config_text["real_time_font_path"]
        self.real_pic_font_size = self.config_text["real_pic_font_size"]
        self.real_pic_font_path = self.config_text["real_pic_font_path"]

        self.real_time_bg_color = self.config_text["real_time_bg_color"]
        self.real_time_font_color = self.config_text["real_time_font_color"]
        self.pic_bg_color = self.config_text["pic_bg_color"]
        self.pic_font_color = self.config_text["pic_font_color"]

        self.pic_count = self.config_text["pic_count"]
        self.real_count = self.config_text["real_count"]


        self.wCam, self.hCam = self.local_configs["camera_settings"]["wCam"],self.local_configs["camera_settings"]["hCam"]
        # 视频这个设置了也不不一定会显示，但是可以斟酌一下 TODO 这个是对于每种摄像头不太一样的设置，我自己的是720，1280.没有办法改变了。但是那个是1080，1920的
        self.get_cam_w,self.get_cam_h = None,None
        self.camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 默认使用内置摄像头
        self.camera.set(3, self.wCam)
        self.camera.set(4, self.hCam)

        self.src_factorx, self.src_factory = self.local_configs['people_settings']['src_factorx'],self.local_configs['people_settings']['src_factory']
        self.position_x, self.position_y = self.local_configs['people_settings']['position_x'],self.local_configs['people_settings']['position_y']
        #  关于手势识别的获得img的裁剪
        # 整体图像裁剪，（1920，1080）--》（y，x）
        self.hand_img_crop_h = self.local_configs["hand_settings"]["hand_img_crop_h"]
        self.hand_img_crop_w = self.local_configs["hand_settings"]["hand_img_crop_w"]

        self.palm_x,self.palm_y =0,0
        # 关于检测范围的测定，可以有选择，来更好的映射与整个屏幕，也可以不选择
        self.pt1, self.pt2 = (200, 100), (400, 300)  # 虚拟鼠标的移动范围，左上坐标pt1，右下坐标pt2
        self.palm_prev_closed = False
        # 关于背景替换的img裁剪
        # 同样是（1920，1080）---》
        self.people_img_crop_h = self.local_configs["people_settings"]["people_img_crop_h"]
        self.people_img_crop_w = self.local_configs["people_settings"]["people_img_crop_w"]
        self.is_show_after = False
        self.once_incv = False
        self.caping = False
        self.fingerstotakepic = False
        # 初始化
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        self.real_counter_font_color = self.local_configs["font_settings"]["counter_font_color"]
        self.picture_font_color = self.local_configs["font_settings"]["picture_font_color"]
        #字体设置
        self.ui.label_real_word.setAlignment(Qt.AlignCenter)
        self.ui.label_real_word.setStyleSheet("background-color:"+self.real_time_bg_color+";color:"+self.real_time_font_color)
        self.ui.label_3.setAlignment(Qt.AlignCenter)
        self.ui.label_3.setFont(QFont("宋体", 40, QFont.Bold))
        self.ui.label_3.setStyleSheet("color:"+self.real_counter_font_color)
        self.ui.label_page4_counter.setAlignment(Qt.AlignCenter)
        self.ui.label_page4_counter.setFont(QFont("宋体", 40, QFont.Bold))
        self.ui.label_page4_counter.setStyleSheet("color:"+self.real_counter_font_color)

        # 连接按钮的点击事件
        # ----------------------------------------------------------------------
        # 页面跳转 page1：2,开始界面  page2：1, 拍照的界面 ， page3：0，选择界面

        self.ui.pushButton_goto_pic.clicked.connect(self.gotopage3)#page1 去拍照
        self.ui.pushButton_setting.clicked.connect(self.gotosetting)
        # self.ui.pushButton_page2_back_1.clicked.connect(self.gotopage1)
        # self.ui.pushButton_change_bg_2.clicked.connect(self.gotopage3) #page2 背景
        # page3
        self.ui.pushButton_page3_back1.clicked.connect(self.gotopage1)  #page2 返回

        # self.ui.pushButton_page3_back2.clicked.connect(self.gotopage4) #page2 ×

        # self.ui.pushButton_change_bg.clicked.connect(self.gotopage2)
        self.ui.pushButton_show_back.clicked.connect(self.gotopage3)
        self.ui.stackedWidget_background.setCurrentIndex(2)
        self.in_page = 0
        # 调试
        #peopele factor
        self.ui.pushButton_p_a.clicked.connect(self.f_a)
        self.ui.pushButton_p_r.clicked.connect(self.f_l)
        self.ui.pushButton_ph_a.clicked.connect(self.p_h_a)
        self.ui.pushButton_ph_r.clicked.connect(self.p_h_l)
        self.ui.pushButton_pw_a.clicked.connect(self.p_w_a)
        self.ui.pushButton_pw_r.clicked.connect(self.p_w_l)
        self.ui.pushButton_ch_a.clicked.connect(self.c_h_a)
        self.ui.pushButton_ch_r.clicked.connect(self.c_h_l)
        self.ui.pushButton_cw_a.clicked.connect(self.c_w_a)
        self.ui.pushButton_cw_r.clicked.connect(self.c_w_l)
        self.ui.pushButton_hh_a.clicked.connect(self.h_h_a)
        self.ui.pushButton_hh_r.clicked.connect(self.h_h_l)
        self.ui.pushButton_hw_a.clicked.connect(self.h_w_a)
        self.ui.pushButton_hw_r.clicked.connect(self.h_w_l)

        self.ui.pushButton_hf_a.clicked.connect(self.h_f_a)
        self.ui.pushButton_hf_r.clicked.connect(self.h_f_l)

        self.ui.label_11.setText(f"{self.people_img_crop_h}")
        self.ui.label_13.setText(f"{self.people_img_crop_w}")
        # self.ui.label_27.setText(f"{self.hand_img_crop_h}")
        # self.ui.label_29.setText(f"{self.hand_img_crop_w}")
        self.ui.checkBox.setChecked(True)
        self.ui.checkBox.stateChanged.connect(self.usehand_box)
        self.is_use_hand = True
        self.hand_factor = self.local_configs["hand_settings"]["hand_factor"]
        self.hand_pos_x = self.local_configs["hand_settings"]["hand_pos_x"]
        self.hand_pos_y = self.local_configs["hand_settings"]["hand_pos_y"]
        self.setting_in = False
        self.setting_save = False
        self.setting_cancle = False
        self.setting_hand = False
        self.ui.pushButton.clicked.connect(self.save_settings)
        self.ui.pushButton_2.clicked.connect(self.cancle_settings)

        self.hand_distance_normal = 0
        self.gesture_duration = 0
        self.gesture_shape = None
        #---------------------------------
        self.ui.checkBox_2.setChecked(False)
        self.ui.checkBox_2.stateChanged.connect(self.checkboxChanged_2) #TODO jinriyaowen
        #---------------------------------
        #拍照的程序
        # TODO page3 和 page2 都可以拍照
        # self.ui.pushButton_page3_take.clicked.connect(self.open_image)
        self.ui.pushButton_show_take.clicked.connect(self.capture_frame)
        self.ui.pushButton_show_take_real.clicked.connect(self.capture_frame)
        self.ui.pushButton_show_back_real.clicked.connect(self.gotopage3)
        # self.ui.pushButton_page2_change.clicked.connect(self.gotopage3)
        # self.ui.pushButton.clicked.connect(self.capture_frame)
        #--------------------------------------------------------------

        # 页面一的操作。只有去拍照已经写完
        # 页面二的操作，有切换背景，去到 page3，写完；回去page1写完。此时点击拍照不可用
        # 页面三的操作，
        # 1.背景的选择 很多的按钮，是stackwidget
        # 2.背景选择的同时，更换背景
        # 3.拍照按钮，用来替换背景，同时显示倒计时
        #---------其中的背景选择按钮设置----------------------------------------------------
        # 1.Create a button group
        self.button_group_pic = QButtonGroup(self)
        self.button_group_pic.setExclusive(True)  # Only one button can be checked at a time
        self.buttons_pic = [
            self.ui.pushButton_pic1_1, self.ui.pushButton_pic1_2, self.ui.pushButton_pic1_3,
            self.ui.pushButton_pic1_4, self.ui.pushButton_pic1_5, self.ui.pushButton_pic1_6,
            self.ui.pushButton_pic2_7, self.ui.pushButton_pic2_8, self.ui.pushButton_pic2_9,
            self.ui.pushButton_pic2_10, self.ui.pushButton_pic2_11, self.ui.pushButton_pic2_12,
            self.ui.pushButton_pic3_13, self.ui.pushButton_pic3_14, self.ui.pushButton_pic3_15,
            self.ui.pushButton_pic3_16, self.ui.pushButton_pic3_17, self.ui.pushButton_pic3_18
        ]
        # Connect each button to the handle_button_click method
        for button in self.buttons_pic:
            button.setCheckable(True)
            button.clicked.connect(self.handle_button_click)
            self.button_group_pic.addButton(button)
        # print(self.buttons)
        # 设置初始背景，应该不用了


        # 页面3-----------------------三大地方选择显示-------------------------------------
        self.ui.stackedWidget_2.setCurrentIndex(0)

        self.button_group_choose = QButtonGroup(self)
        self.button_group_choose.setExclusive(True)  # Only one button can be checked at a time
        self.buttons_choose = [
            self.ui.pushButton_old, self.ui.pushButton_school, self.ui.pushButton_building,
        ]
        for button_c in self.buttons_choose:
            # print("buttons")
            button_c.setCheckable(True)
            button_c.clicked.connect(self.handle_button_click_choose)
            self.button_group_choose.addButton(button_c)

        # 时间管理
        # 更新手势的时间,大概20帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # 拍摄计时外初始化
        self.capture_close_timer = QTimer(self)
        self.capture_close_timer.timeout.connect(self.clear_status)
        # 为的是倒计时的计数
        self.capture_countdown_timer = QTimer(self)
        self.capture_countdown_timer.timeout.connect(self.capture_countdown_update)
        self.capture_countdown_seconds = 3
        # 结束后0.5s启动
        self.capture_over_timer = QTimer(self)
        self.capture_over_timer.timeout.connect(self.capture_show_over)
        # qrcode 初始化
        self.qr_close_timer = QTimer(self)
        self.qr_close_timer.timeout.connect(self.qr_closeWindow)
        # qrcode 倒计时
        self.qr_counter_timer = QTimer(self)
        self.qr_counter_timer.timeout.connect(self.qr_counter_update)
        self.qr_countdown_seconds = 10

        #
        self.new_cap_counter = QTimer(self)
        self.new_cap_counter.timeout.connect(self.capture_countdown_update)

       # 添加一个变量用于保存背景图像路径
       #  self.bg_img_path = r'E:\ALLCODE\Pythoncode\virtual_backgroud\simpleui_cap\data\5.jpg'
        self.bg_img_path = os.path.abspath(os.path.join(base_path, "data/matting_bg/demo2.png"))
        self.config = os.path.abspath(os.path.join(base_path, 'configs/v2-lite-192/deploy.yaml'))
        # print(self.config)
        self.vertical_screen = False
        self.init = True

        self.predictor = Predictor(self.config,self.vertical_screen,)
        self.show_pixel = False
        self.show_real = False
        self.is_showqr = False
        # cap
        self.capture_window = None
        #real
        self.show_real_crop_h = 1080
        self.is_mirror_real = True
        self.show_real = False
        botton_jpg = os.path.join(self.bash_path,'data/pic_fill/botton.jpg')
        botton_png = os.path.join(self.bash_path,'data/pic_fill/botton.png')
        if os.path.isfile(botton_png):
            self.botton_path = botton_png
        else:
            self.botton_path = botton_jpg
        self.show_real_botton_pic = cv2.imread(self.botton_path)
        #matting
        self.get_cfg = os.path.abspath(os.path.join(base_path, 'Matting/configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml'))
        self.image_path = None
        self.model_path = os.path.abspath(os.path.join(base_path,'Matting/pretrained_models/ppmattingv2-stdc1-human_512.pdparams'))
        self.device = 'gpu'
        # about path
        self.real_pic_path = os.path.abspath(os.path.join(base_path,"data/cap/real.png  "))
        self.save_cap_dir = os.path.abspath(os.path.join(base_path,"data/cap"))      # 保存拍摄图片的文件夹
        self.after_pro_dir = os.path.abspath(os.path.join(base_path,"data/after"))     # 保存抠图后的文件夹

        self.save_cap_path = None
        self.after_path = None
        self.qr_path = None

        self.image_count = 1

        # matting pic path
        self.mpic_path = [os.path.abspath(os.path.join(base_path, "data/matting_bg/demo1.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/demo2.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/demo3.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic1_4.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic1_5.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic1_6.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic2_7.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic2_8.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic2_9.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic2_10.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic2_11.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic2_12.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic3_13.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic3_14.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic3_15.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic3_16.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic3_17.png")),
                          os.path.abspath(os.path.join(base_path, "data/matting_bg/mpic3_18.png")),
        ]

        # matting proccess
        self.matting_cfg = Config(self.get_cfg)
        self.matting_builder = MatBuilder(self.matting_cfg)

        self.matting_model = self.matting_builder.model
        self.matting_transforms = ppmatting.transforms.Compose(self.matting_builder.val_transforms)

        #hand
        # self.width_scr , self.height_scr = 1080,760
        self.width_scr, self.height_scr = autopy.screen.size()  # 1280 720 是坐标点的位置，而不是像素点。之后的移动也是根据坐标点

        # print("self.width_scr,self.height_scr", self.width_scr, self.height_scr)
        # 初始化MediaPipe Hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8,
                               min_tracking_confidence=0.5)

        self.detector = HandDetector(staticMode=False,  # 视频流图像
                        maxHands=1,  # 最多检测一只手
                        detectionCon=0.8,  # 最小检测置信度
                        minTrackCon=0.5)  # 最小跟踪置信度

        self.init_count = 0
        self.fingers_count = 0
        self.capture_static = np.zeros((0,0,0))
        self.smooth = 2  # 自定义平滑系数，让鼠标移动平缓一些
        self.start_camera()
        # 添加右键菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)


    def usehand_box(self,state):
        if state == 0:
            self.is_use_hand = False
        else:
            self.is_use_hand = True

    def checkboxChanged_2(self,state):
        if state==0:
            self.show_real = False
        else:
            self.show_real = True
            self.gotopage4()

    def f_a(self):
        if self.get_cam_h is not None and self.get_cam_w is not None:
            max_factorx = round(self.get_cam_w/self.people_img_crop_w,2)
            max_factory = round(self.get_cam_h//self.people_img_crop_h,2)
            max_factor = min(max_factorx,max_factory)
            print(max_factor)
        else:
            max_factor = 1
        self.src_factorx = min(max_factor,round(self.src_factorx+0.1,2))
        self.src_factory = min(max_factor,round(self.src_factory+0.1,2))
        self.ui.label_5.setText(f"{round(self.src_factorx,1)}")
        self.local_configs["people_settings"]["src_factorx"] = self.src_factorx
        self.local_configs["people_settings"]["src_factory"] = self.src_factory

    def f_l(self):
        self.src_factorx = max(0.01,round(self.src_factorx-0.1,2))
        self.src_factory = max(0.01,round(self.src_factory-0.1,2))
        self.ui.label_5.setText(f"{round(self.src_factorx,1)}")
        self.local_configs["people_settings"]["src_factorx"] = self.src_factorx
        self.local_configs["people_settings"]["src_factory"] = self.src_factory

    def h_f_a(self):
        if self.get_cam_h is not None and self.get_cam_w is not None:
            max_factorx = round(self.get_cam_w//self.hand_img_crop_w,2)
            max_factory = round(self.get_cam_h//self.hand_img_crop_h,2)
            max_factor = min(max_factorx,max_factory)
        else:
            max_factor = 1
        self.hand_factor = min(max_factor,round(self.hand_factor+0.1,2))
        self.ui.label_32.setText(f"{round(self.hand_factor,1)}")
        self.local_configs["hand_settings"]["hand_factor"] = self.hand_factor

    def h_f_l(self):
        self.hand_factor = max(0.01,round(self.hand_factor-0.1,2))
        self.ui.label_32.setText(f"{round(self.hand_factor,1)}")
        self.local_configs["hand_settings"]["hand_factor"] = self.hand_factor

    def p_h_a(self):
        self.position_y = self.position_y + 20
        self.ui.label_7.setText(f"{round(self.position_y,1)}")
        self.local_configs["people_settings"]["position_y"] = self.position_y

    def p_h_l(self):
        self.position_y = self.position_y - 20
        self.ui.label_7.setText(f"{round(self.position_y,1)}")
        self.local_configs["people_settings"]["position_y"] = self.position_y

    def p_w_a(self):
        self.position_x = self.position_x + 20
        self.ui.label_9.setText(f"{round(self.position_x,1)}")
        self.local_configs["people_settings"]["position_x"] = self.position_x

    def p_w_l(self):
        self.position_x = self.position_x - 20
        self.ui.label_9.setText(f"{round(self.position_x,1)}")
        self.local_configs["people_settings"]["position_x"] = self.position_x

    def c_h_a(self):
        if self.get_cam_h is not None:
            max_h = self.get_cam_h
        else:
            max_h = self.hCam

        self.people_img_crop_h = min(max_h, self.people_img_crop_h + 10)
        self.ui.label_11.setText(f"{round(self.people_img_crop_h,1)}")
        self.local_configs["people_settings"]["people_img_crop_h"] = self.people_img_crop_h

    def c_h_l(self):
        self.people_img_crop_h = max(1, self.people_img_crop_h - 10)
        self.ui.label_11.setText(f"{round(self.people_img_crop_h,1)}")
        self.local_configs["people_settings"]["people_img_crop_h"] = self.people_img_crop_h

    def c_w_a(self):
        if self.get_cam_w is not None:
            max_w = self.get_cam_w
        else:
            max_w = self.wCam

        self.people_img_crop_w = min(max_w, self.people_img_crop_w + 10)
        self.ui.label_13.setText(f"{round(self.people_img_crop_w,1)}")
        self.local_configs["people_settings"]["people_img_crop_w"] = self.people_img_crop_w

    def c_w_l(self):
        self.people_img_crop_w = max(1, self.people_img_crop_w -10)
        self.ui.label_13.setText(f"{round(self.people_img_crop_w,1)}")
        self.local_configs["people_settings"]["people_img_crop_w"] = self.people_img_crop_w

    def h_h_a(self):
        if self.get_cam_h is not None:
            max_h = self.get_cam_h
        else:
            max_h = self.hCam
        self.hand_pos_y = min(max_h, self.hand_pos_y + 10)
        self.ui.label_27.setText(f"{round(self.hand_pos_y,1)}")
        self.local_configs["hand_settings"]["hand_pos_y"] = self.hand_pos_y

    def h_h_l(self):
        self.hand_pos_y = max(1, self.hand_pos_y - 10)
        self.ui.label_27.setText(f"{round(self.hand_pos_y,1)}")
        self.local_configs["hand_settings"]["hand_pos_y"] = self.hand_pos_y

    def h_w_a(self):
        if self.get_cam_w is not None:
            max_w = self.get_cam_w
        else:
            max_w = self.wCam

        self.hand_pos_x = min(max_w, self.hand_pos_x + 10)
        self.ui.label_29.setText(f"{round(self.hand_pos_x,1)}")
        self.local_configs["hand_settings"]["hand_pos_x"] = self.hand_pos_x

    def h_w_l(self):
        self.hand_pos_x = max(1, self.hand_pos_x - 10)
        self.ui.label_29.setText(f"{round(self.hand_pos_x,1)}")
        self.local_configs["hand_settings"]["hand_pos_x"] = self.hand_pos_x


    def read_mytext(self,text_file):
        '''
        txt中的数据读取
        chinese_text，
        font_path这个是下载图片的字体
        font_size这个是下载图片的字体大小
        font这个是实时显示时的字体
        font_size
        '''
        with open(text_file, "r", encoding="utf-8") as file:
            chinese_texts = [line.strip() for line in file]
            chinese_text = chinese_texts[0].split(",")[1]
            font_path_pic = chinese_texts[1].split(",")[1]
            font_size_pic = chinese_texts[2].split(",")[1]
            font_path_real = chinese_texts[3].split(",")[1]
            font_size_real = chinese_texts[4].split(",")[1]
            detector = chinese_texts[5].split(",")[1]

        return {"chinese":chinese_text,
                "font_path_pic":font_path_pic,"font_size_pic":font_size_pic,
                "font_path_real":font_path_real,"font_size_real":font_size_real,
                "hand_mp_cvzone": detector
            }

    def get_name_bg_matting(self):
        bg_dir = os.path.abspath(os.path.join(self.base_path, "data/matting_bg/"))
        image_files = []
        for root, dirs, files in os.walk(bg_dir):
            for file in files:
                if file.endswith(('.png', '.jpg')) and 'mpic' in file:
                    name = file.split('_')[1].split('.')[0]
                    image_files.append(name)
                    image_name = set(image_files)
        return image_name

    def handle_button_click(self):
        # Handle button click
        clicked_button = self.sender()
        for button in self.buttons_pic:
            if button is clicked_button and button.isChecked():
                print(f'{button.objectName()} pressed')
                self.set_button_background_color(button)
                self.set_blackbg(button)
                self.gotopage2()
                # self.new_cap_counter.start(3000)
            else:
                button.setChecked(False)

    def handle_button_click_choose(self):
        # Handle button click
        clicked_button_c = self.sender()
        for button_c in self.buttons_choose:
            if button_c is clicked_button_c and button_c.isChecked():
                print(f'{button_c.objectName()} pressed')
                botton_name = button_c.objectName() # Extract button index from objectName
                if botton_name == "pushButton_old":
                    self.ui.stackedWidget_2.setCurrentIndex(0)
                elif botton_name == "pushButton_school":
                    self.ui.stackedWidget_2.setCurrentIndex(1)
                elif botton_name == "pushButton_building":
                    self.ui.stackedWidget_2.setCurrentIndex(3)
                else:
                    self.ui.stackedWidget_2.setCurrentIndex(0)
            else:
                button_c.setChecked(False)

    def set_button_background_color(self, button):
        # Set different background colors for different buttons
        button_name = button.objectName()
        button_index = int(button.objectName().split('_')[-1])  # Extract button index from objectName
        pic_list_len = len(self.mpic_path)
        if button_index <= pic_list_len:
            self.set_bg_img_path(self.mpic_path[button_index-1])
        else:
            self.set_bg_img_path(self.mpic_path[pic_list_len-1])

    def set_blackbg(self,button):
        button_str = str(button.objectName().split('_')[-1])  # Extract button index from objectName
        style_str1 = "#stackedWidget_backgroundPage1{border-image: url(:/bg_select/icon/bg/selected/demo" + button_str + ".png);}"
        self.ui.stackedWidget_backgroundPage1.setStyleSheet(style_str1)

    def gotosetting(self):
        self.ui.stackedWidget_background.setCurrentIndex(4)
        self.show_pixel = True
        self.show_real = False
        self.setting_in = True

    def save_settings(self,path):
        save_json_after(self.json_path,self.local_configs)
        self.setting_hand = False
        self.setting_in = False
        self.ui.stackedWidget_background.setCurrentIndex(2)

    def cancle_settings(self):
        self.setting_hand = False
        self.setting_in = False
        self.ui.stackedWidget_background.setCurrentIndex(2)

    def gotopage2(self):
        self.ui.stackedWidget_background.setCurrentIndex(1)
        # self.ui.pushButton_5.setCheckable(False)
        self.show_pixel = True
        self.show_real = False
        self.in_page = 2

    def gotopage4(self):
        self.ui.stackedWidget_background.setCurrentIndex(3)
        self.show_pixel = False
        self.show_real = True
        # self.ui.checkBox_2.setChecked(False)
        self.ui.checkBox_2.toggle()#从开启到关闭
        self.setting_in = False
        self.in_page = 4

    def gotopage1(self):
        self.ui.stackedWidget_background.setCurrentIndex(2)
        self.show_pixel = False
        self.show_real = False
        self.in_page = 1
    # 页面三基本显示操作

    def gotopage3(self):
        self.ui.stackedWidget_background.setCurrentIndex(0)
        # self.ui.pushButton_5.setCheckable(True)
        # self.ui.pushButton_change_bg.setCheckable(False)
        self.show_pixel = False
        self.show_real = False
        self.in_page = 3

    def show_page3(self):
        self.ui2.stackedWidget_background.setCurrentIndex(0)

    def start_camera(self):
        self.timer.start(20)  # 30 fps
        self.update_frame()

    def pause_camera(self):
        self.timer.stop()

    def detect_hand_gesture(self, hand_landmarks):
        # 计算手指之间的距离
        thumb_tip = np.array([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x,
                              hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y])
        index_finger_tip = np.array([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x,
                                     hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y])
        middle_finger_tip = np.array([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                      hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y])
        ring_finger_tip = np.array([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x,
                                    hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y])
        pinky_tip = np.array([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x,
                              hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y])

        # 计算手指之间的距离
        thumb_index_distance = np.linalg.norm(thumb_tip - index_finger_tip)
        index_middle_distance = np.linalg.norm(index_finger_tip - middle_finger_tip)
        middle_ring_distance = np.linalg.norm(middle_finger_tip - ring_finger_tip)
        ring_pinky_distance = np.linalg.norm(ring_finger_tip - pinky_tip)

        # 检查手势是否为拳头
        if thumb_index_distance < self.local_configs["hand_settings"]["distance"] and index_middle_distance < self.local_configs["hand_settings"]["distance"] \
                and middle_ring_distance < self.local_configs["hand_settings"]["distance"] and ring_pinky_distance < self.local_configs["hand_settings"]["distance"]:
            return True
        else:
            return False

    def visualize_hand_landmarks(self , image, hand_landmarks, gesture):
        mp_drawing = mp.solutions.drawing_utils
        # 设置绘制手部关键点和连接线的颜色
        if gesture:
            drawing_color = (0, 255, 0)  # 如果检测到拳头，使用绿色
        else:
            drawing_color = (255, 0, 0)  # 如果未检测到拳头，使用蓝色

        # 绘制手部关键点
        mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=drawing_color, thickness=2,
                                                                               circle_radius=4))

    def update_frame(self):
        if self.is_showqr:
            self.ui.pushButton_show_take.setEnabled(False)
            self.ui.pushButton_show_back.setEnabled(False)
            self.ui.pushButton_show_back_real.setEnabled(False)
            self.ui.pushButton_show_take_real.setEnabled(False)
        else:
            self.ui.pushButton_show_take.setEnabled(True)
            self.ui.pushButton_show_back.setEnabled(True)
            self.ui.pushButton_show_back_real.setEnabled(True)
            self.ui.pushButton_show_take_real.setEnabled(True)
        cap_camera = self.camera
        assert cap_camera.isOpened(), "Fail to open camera"
        ret, img_o = self.camera.read()
        self.get_cam_h,self.get_cam_w = img_o.shape[0],img_o.shape[1]
        #我自己的电脑h720 w1280，应该是最大的了
        # print("get hand image shape:h:%d,w:%d" % (img_o.shape[0], img_o.shape[1]))
        # start_time = time.time()
        if self.in_page == 2 or self.in_page ==4 :
            crop_w = self.hand_img_crop_w * self.hand_factor
            crop_h = self.hand_img_crop_h * self.hand_factor
            img = self.crop_top(img_o, crop_w, crop_h, self.hand_pos_x, self.hand_pos_y)
            hands, _ = self.detector.findHands(img,draw=False, flipType=False)  # 上面反转过了，这里就不用再翻转了
        else:
            hands = False
        if ret and self.show_real is False:
            if True:
                if False:
                    results = self.mp_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # results = self.mp_detector.process(img)
                    # mediapip如果能检测到手那么就进行下一步
                    if results.multi_hand_landmarks and (self.setting_in is False or self.setting_hand == True) and self.is_showqr is False:
                        # print("iamge",img.shape)
                        hand_landmarks = results.multi_hand_landmarks[0]
                        if self.setting_hand:
                            self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        # 计算食指和拇指指尖之间的距离
                        index_finger_tip = hand_landmarks.landmark[7].y
                        index_finger_tip1 = hand_landmarks.landmark[8].y

                        # distance = np.math.sqrt(
                            # (index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)
                        if self.init_count == 0:
                            palmfuncx = 0
                            palmfuncy = 0
                            self.init_count = self.init_count + 1
                        # 计算手掌中心的坐标
                        for index in [0, 3, 5, 9, 13, 17]:
                            landmark = hand_landmarks.landmark[index]
                            self.palm_x += landmark.x
                            self.palm_y += landmark.y

                        self.palm_x /= 6
                        self.palm_y /= 6

                        # 将坐标转换为屏幕上的位置
                        palm_x = int(self.palm_x * self.width_scr)
                        palm_y = int(self.palm_y * self.height_scr)
                        # 边界检查，确保不超出屏幕范围
                        palm_x = min(max(palm_x, 2), self.width_scr-10)
                        palm_y = min(max(palm_y, 2), self.height_scr-10)
                        # 移动鼠标

                        # 如果距离小于一定阈值，则认为是握拳
                        fist = self.detect_hand_gesture(hand_landmarks)
                        if fist is False and index_finger_tip1<index_finger_tip:
                            autopy.mouse.move(palm_x, palm_y)
                        else:
                            pass

                        # fist = distance < self.local_configs["hand_settings"]["distance"]
                        if self.setting_hand:
                            # print("拇指的食指的距离(用于判断）",distance)
                            # print("手腕和小指的距离（标准）", self.hand_distance_normal)
                            # print("他们之间的比例关系",round(distance/self.hand_distance_normal,2))
                            print("fist",fist)
                            self.visualize_hand_landmarks(img, hand_landmarks, fist)

                        # 如果上一帧未闭合而当前帧闭合，则触发点击事件
                        if fist:
                            self.gesture_duration += 1
                            if self.gesture_shape is None:
                                self.gesture_shape = 'fist'
                        else:
                            self.gesture_duration = 0
                            self.gesture_shape = None

                        # 如果在时间窗口内持续检测到握拳手势且手势形状符合条件，则触发点击事件
                        if self.gesture_duration >= self.local_configs["hand_settings"]["duration_threshold"] and self.gesture_shape == 'fist' and self.palm_prev_closed is False:
                            # 在这里可以添加更多条件，例如手势形状等
                            autopy.mouse.click()
                            self.palm_prev_closed = True
                        else:
                            self.palm_prev_closed = False

                    if self.setting_hand == True:
                        cv2.imshow("hand control", img)
                        cv2.setWindowProperty("hand control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.camera.release()
                            cv2.destroyAllWindows()
                            once_incv = True
                    if self.setting_hand is False and self.once_incv:
                        cv2.destroyAllWindows()
                        self.once_incv = False
                if hands and self.is_use_hand:
                    fingers = self.detector.fingersUp(hands[0])  # 传入
                    # print(fingers)
                    if fingers[0] == 0 and fingers[1] == 1 and fingers[2]==1 and fingers[3]==1 and fingers[4]==1:
                        self.fingers_count +=1
                        print(self.fingers_count)
                        if self.fingers_count == self.pic_count:
                            self.fingerstotakepic = True
                            self.fingers_count=0
                    else:
                        self.fingerstotakepic = False
                else:
                    self.fingerstotakepic = False

                if self.show_pixel and self.setting_hand is False and self.caping is False:
                    # img = cv2.flip(img, flipCode=1)  # 1代表水平翻转，0代表竖直翻转
                    img_pixel = self.crop_di(img_o, self.people_img_crop_w, self.people_img_crop_h)
                    if self.setting_in:
                        print("")
                        p = os.path.join(self.bash_path,"data/matting_bg/white.png")
                        bg = get_bg(p)
                    else:
                        bg = get_bg(self.bg_img_path)
                    self.image_predict_thread = Image_predict_Thread(img_pixel, bg, self.src_factorx,self.src_factory,self.position_x,self.position_y,self.predictor)
                    self.image_predict_thread.predict_finished.connect(self.handle_processing_finished)
                    self.image_predict_thread.start()

                    # # 在主线程中等待线程完成
                    # self.image_predict_thread.wait()
                    # # 删除QThread对象
                    # self.image_predict_thread.deleteLater()
                    # out = self.predictor.run(img, bg, self.src_factorx, self.src_factory, self.position_x,
                    #                              self.position_y)
                    # self.capture_static = out
                    # h, w, c = out.shape
                    # # print(out.shape)
                    # q_image = QImage(out.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                    # pixmap = QPixmap.fromImage(q_image)
                    # if self.setting_in == True and self.setting_hand is False:
                    #     self.ui.label_settingshow.setPixmap(pixmap)
                    #
                    # elif self.is_show_after is False:
                    #     self.ui.label_show.setPixmap(pixmap)

        if ret and self.show_real:
            if hands and self.is_use_hand:
                # 获取手部信息hands中的21个关键点信息
                # lmList = hands[0]['lmList']  # hands是由N个字典组成的列表，字典包每只手的关键点信息
                fingers = self.detector.fingersUp(hands[0])  # 传入
                if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                    self.fingers_count += 1
                    print(self.fingers_count)
                    if self.fingers_count == self.real_count:
                        self.fingerstotakepic = True
                        self.fingers_count = 0
                else:
                    self.fingerstotakepic = False
            else:
                self.fingerstotakepic = False

            # img = cv2.flip(img, flipCode=1)  # 1代表水平翻转，0代表竖直翻转
            img_real = self.crop_di(img_o, self.show_real_crop_h, self.show_real_crop_h).copy()
            if self.is_mirror_real:
                img_real = cv2.flip(img_real, flipCode=1)
            text_path = os.path.join(self.bash_path,'text.txt')
            # botton_path = os.path.join(self.bash_path,'data/pic_fill/botton.png')
            new_l, text_info = util.show_real(text_path, img_real, self.botton_path)
            # cv2.imwrite(self.real_pic_path,new_l)

            self.ui.label_real_word.setFont(QFont(str(self.real_time_font_path), int(self.real_time_font_size), QFont.Bold))
            self.ui.label_real_word.setText(self.welcome_words_in_realtime_show)
            # 显示图像
            h, w, c = new_l.shape
            # print(out.shape)
            q_image = QImage(new_l.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.ui.label_real.setPixmap(pixmap)
            # 图片显示
            h1,w1,c1 = self.show_real_botton_pic.shape
            q_image1 = QImage(self.show_real_botton_pic.data, w1, h1, w1 * c1, QImage.Format_RGB888).rgbSwapped()
            pixmap1 = QPixmap.fromImage(q_image1)
            self.ui.label_real_2.setPixmap(pixmap1)
            if self.is_show_after:
                cap = cv2.imread(self.after_path)
                # 显示图像
                h, w, c = cap.shape
                # print(out.shape)
                q_image = QImage(cap.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                self.ui.label_real.setPixmap(pixmap)
                # 图片显示
                h1,w1,c1 = self.show_real_botton_pic.shape
                q_image1 = QImage(self.show_real_botton_pic.data, w1, h1, w1 * c1, QImage.Format_RGB888).rgbSwapped()
                pixmap1 = QPixmap.fromImage(q_image1)
                self.ui.label_real_2.setPixmap(pixmap1)

        if self.fingerstotakepic and (self.in_page==2 or self.in_page==4) and self.caping is False and self.is_showqr is False:
            self.capture_frame()

    def handle_processing_finished(self,predict_img):
        out = cv2.cvtColor(predict_img, cv2.COLOR_BGR2RGB)
        self.capture_static = out
        h, w, c = out.shape
        # print(out.shape)
        q_image = QImage(out.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        if self.setting_in == True and self.setting_hand is False:
            self.ui.label_settingshow.setPixmap(pixmap)

        elif self.is_show_after is False:
            self.ui.label_show.setPixmap(pixmap)

    def capture_frame(self):
        self.fingers_count = 0
        # Show countdown and start the countdown timer
        # 倒计时开始就关闭手势识别
        self.is_showqr = True
        if self.is_showqr:
            self.ui.pushButton_show_take.setEnabled(False)
            self.ui.pushButton_show_back.setEnabled(False)
            self.ui.pushButton_show_back_real.setEnabled(False)
            self.ui.pushButton_show_take_real.setEnabled(False)
        else:
            self.ui.pushButton_show_take.setEnabled(True)
            self.ui.pushButton_show_back.setEnabled(True)
            self.ui.pushButton_show_back_real.setEnabled(True)
            self.ui.pushButton_show_take_real.setEnabled(True)
        self.capture_countdown_seconds = 6
        if self.show_real:
            self.ui.label_page4_counter.setText(f"倒计时: 5秒")
        else:
            self.ui.label_3.setText(f"倒计时: 5秒")
        self.capture_countdown_timer.start(1000)

    def capture_countdown_update(self):
        self.fingers_count = 0
        if self.is_showqr:
            self.ui.pushButton_show_take.setEnabled(False)
            self.ui.pushButton_show_back.setEnabled(False)
            self.ui.pushButton_show_back_real.setEnabled(False)
            self.ui.pushButton_show_take_real.setEnabled(False)
        else:
            self.ui.pushButton_show_take.setEnabled(True)
            self.ui.pushButton_show_back.setEnabled(True)
            self.ui.pushButton_show_back_real.setEnabled(True)
            self.ui.pushButton_show_take_real.setEnabled(True)
        self.capture_countdown_seconds -= 1
        # 之前遇到会卡在1处，很奇怪不懂为什么
        if self.capture_countdown_seconds != 1:
            if self.show_real:
                self.ui.label_page4_counter.setText(f"倒计时: {self.capture_countdown_seconds-1}秒")
            else:
                self.ui.label_3.setText(f"倒计时: {self.capture_countdown_seconds-1}秒")
        # 当变成1的时候，再启动一个倒计时
        if self.capture_countdown_seconds == 2:
            self.capture_over_timer.start(750)

        if self.capture_countdown_seconds == 1:
            self.capture_frame_after_countdown()

    def capture_over_static(self,img_path):
        img = cv2.imread(img_path)
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = out.shape
        # print(out.shape)
        q_image = QImage(out.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.label_show.setPixmap(pixmap)

    def capture_over_static_real(self,img_path,text_path):
        self.fingers_count = 0

        img = cv2.imread(img_path)
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = out.shape
        # print(out.shape)
        q_image = QImage(out.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.label_real.setPixmap(pixmap)

    def capture_show_over(self):
        self.fingers_count = 0

        if self.is_showqr:
            self.ui.pushButton_show_take.setEnabled(False)
            self.ui.pushButton_show_back.setEnabled(False)
            self.ui.pushButton_show_back_real.setEnabled(False)
            self.ui.pushButton_show_take_real.setEnabled(False)
        else:
            self.ui.pushButton_show_take.setEnabled(True)
            self.ui.pushButton_show_back.setEnabled(True)
            self.ui.pushButton_show_back_real.setEnabled(True)
            self.ui.pushButton_show_take_real.setEnabled(True)
        self.capture_over_timer.stop()
        if self.show_real:
            self.ui.label_page4_counter.setText(f"请等待")
        else:
            self.ui.label_3.setText("请等待")
        self.capture_close_timer.start(1000)  # 3秒后擦除状态文本

    def capture_frame_after_countdown(self):
        self.fingers_count = 0
        self.caping = True
        if self.is_showqr:
            self.ui.pushButton_show_take.setEnabled(False)
            self.ui.pushButton_show_back.setEnabled(False)
            self.ui.pushButton_show_back_real.setEnabled(False)
            self.ui.pushButton_show_take_real.setEnabled(False)
        else:
            self.ui.pushButton_show_take.setEnabled(True)
            self.ui.pushButton_show_back.setEnabled(True)
            self.ui.pushButton_show_back_real.setEnabled(True)
            self.ui.pushButton_show_take_real.setEnabled(True)
        # self.capture_over_static(self.capture_static)
        self.capture_countdown_timer.stop()
        ret, frame = self.camera.read()
        #对于目标机是h，w（1920，1080）
        # 我这里是h，w（720，1280）
        # 同样需要裁剪中间的
        # cv2.imwrite("get frame.jpg",frame)
        # print("capture get crop size:h:%d,w:%d" % (frame.shape[1],frame.shape[0]))
        if self.show_real:
            frame = self.crop_di(frame, self.show_real_crop_h, self.show_real_crop_h)
        else:
            frame = self.crop_di(frame,self.people_img_crop_w,self.people_img_crop_h)
        # cv2.imwrite("first_resize.jpg", frame)
        # print("capture after crop size:h:%d,w:%d" % (frame.shape[1],frame.shape[0]))
        save_img_filename = f'captured_frame_{self.image_count}.jpg'
        after_path_filename = f'after_{self.image_count}.jpg'
        qr_filename = f'qr_{self.image_count}.jpg'
        self.save_cap_path = os.path.join(self.save_cap_dir, save_img_filename)
        self.after_path = os.path.join(self.after_pro_dir, after_path_filename)
        self.qr_path = os.path.join(self.after_pro_dir, qr_filename)

        self.image_count += 1
        if self.image_count > 5:
            self.image_count = 1  # Reset the count if it exceeds 10
            self.delete_pic(self.save_cap_dir)
            self.delete_pic(self.after_pro_dir)

        if ret and self.show_real==False:
            # 把当前的照片存在save_cap_path中
            cv2.imwrite(self.save_cap_path, frame)
            alpha, fg = predict(
                self.matting_model,
                model_path=self.model_path,
                transforms=self.matting_transforms,
                image_list=[self.save_cap_path],
                trimap_list=[None],
                save_dir=self.after_pro_dir,
                fg_estimate=True)
            # 抠图
            bg = get_bg(self.bg_img_path)
            # 重新读取，其实也可以不读取
            # img_ori = cv2.imread(self.save_cap_path)
            img_ori = frame.copy()
            i_h, i_w, i_c = img_ori.shape  # (h,w,c)
            # bg
            bg_h, bg_w, bg_c = bg.shape
            # print("bg_shape",bg_h,bg_w)
            # 新的alpha图层大小（1920，1080）
            newalpha_layer = np.zeros((bg_h, bg_w), dtype=np.uint8)
            #看看之前的alpha图层大小是多少就是裁剪后的
            alpha_height, alpha_width = alpha.shape
            # print("predict alpha size h:%d,w%d" % (alpha_height,alpha_width))
            # 缩放一下
            resize_a = cv2.resize(alpha, (int(alpha_width * self.src_factorx), int(alpha_height * self.src_factorx)))
            resize_h, resize_w = resize_a.shape
            # print("resize shape",resize_h,resize_w)

            # 计算将小图像放在黑色图层中央的位置
            self.position_y = max(0, min(self.position_y, int(bg_h - resize_h-1)))
            start_y = int(bg_h - resize_h - self.position_y)
            end_y = start_y + resize_h
            x = max(0, min(np.abs(self.position_x), int(bg_w - resize_w-1)))
            if self.position_x > 0:
                start_x = int((bg_w - resize_w) / 2 + x)
            else:
                start_x = int((bg_w - resize_w) / 2 - x)
            end_x = start_x + resize_w
            # 得到了新的alpha图层
            newalpha_layer[start_y:end_y, start_x:end_x] = resize_a
            # cv2.imwrite("new_alpah.jpg", newalpha_layer)

            newrgba_layer = np.ones((bg.shape), dtype=np.uint8) * 255
            rgba_resize = cv2.resize(fg, (int(alpha_width * self.src_factorx), int(alpha_height * self.src_factory)))
            # #得到了新的图层
            newrgba_layer[start_y:end_y, start_x:end_x, :] = rgba_resize
            newalpha_layer = newalpha_layer / 255.0
            newalpha_layer = newalpha_layer[:, :, np.newaxis]
            com = newalpha_layer * newrgba_layer + (1 - newalpha_layer) * bg
            cv2.imwrite(self.after_path, com)
            self.capture_over_static(self.after_path)
            # 抠图完成存好
            # self.merge_matting_pic(self.after_path,self.after_path)

            # 然后返回到界面3 ，并且显示
            # self.gotopage3()

            util.upload(self.after_path, self.qr_path)
            # self.show_captured_image()
            self.caping = False
            self.show_qrcode(self.qr_path)

        elif ret and self.show_real==True:
            #直接拼图把上面的图片和下面的风景拼接起来
            if self.is_mirror_real:
                frame = cv2.flip(frame, flipCode=1)
            cv2.imwrite(self.save_cap_path, frame)
            # 1920 1080 3
            text_path = os.path.join(self.bash_path,'text.txt')
            list_to_image=[self.welcome_words_in_pic_show,self.real_pic_font_path,self.real_pic_font_size]
            show_frame = util.create_final_image(list_to_image,self.save_cap_path,self.botton_path,self.after_path,self.pic_font_color,self.pic_bg_color)
            # 抠图完成存好
            # cv2.imwrite(self.after_path, show_frame)
            util.upload(self.after_path,self.qr_path)
            self.capture_over_static_real(self.after_path,text_path)
            self.caping = False
            self.show_qrcode(self.qr_path)

    def merge_matting_pic(self,image_path,save_path):
        image = Image.open(image_path).convert("RGB")

        logo = Image.open("./data/pic_fill/logo.png")
        # 获取图片的宽度和高度
        width, height = image.size
        # 获取logo的宽度和高度
        logo_width, logo_height = logo.size
        # 设置logo的位置（右上角）
        position = (width - logo_width, 0)
        # 叠加logo到图片上
        image.paste(logo, position, logo)
        # 保存处理后的图片
        image.save(save_path)

    def resize_after_predict(self, alpha, bg, fg, fac_x, fac_y, pos_x, pos_y):
        '''
        fac_x:x方向缩放的大小设置
        fac_y:y方向缩放的大小设置
        posx：x方向移动
        posy：y方向移动
        '''
        bg_h, bg_w, bg_c = bg.shape
        print("bg_shape", bg_h, bg_w)
        # 新的alpha图层大小（1920，1080）
        newalpha_layer = np.zeros((bg_h, bg_w), dtype=np.uint8)
        # 看看之前的alpha图层大小是多少
        alpha_height, alpha_width = alpha.shape
        print("predict alpha size h:%d,w%d" % (alpha_height, alpha_width))
        # 缩放一下
        resize_a = cv2.resize(alpha, (int(alpha_width * fac_x), int(alpha_height * fac_y)))
        resize_h, resize_w = resize_a.shape
        print("resize shape", resize_h, resize_w)

        # 计算将小图像放在黑色图层中央的位置
        #  设置移动的位置，初始是在最下面，移动也是向上加从0开始
        y = max(0, min(pos_y, int(bg_h - resize_h)))
        start_y = int(bg_h - resize_h - y)
        end_y = start_y + resize_h

        x = min(np.abs(pos_x), int((bg_w - resize_w)/2) - 1)
        start_x = int((bg_w - resize_w) / 2 + x)
        end_x = start_x + resize_w
        # 得到了新的alpha图层
        newalpha_layer[start_y:end_y, start_x:end_x] = resize_a
        cv2.imwrite("new_alpah.jpg", newalpha_layer)

        newrgba_layer = np.ones((bg.shape), dtype=np.uint8) * 255
        rgba_resize = cv2.resize(fg, (int(alpha_width * fac_x), int(alpha_height * fac_y)))
        # #得到了新的图层
        newrgba_layer[start_y:end_y, start_x:end_x, :] = rgba_resize
        cv2.imwrite("new_rgba.jpg", newrgba_layer)
        newalpha_layer = newalpha_layer / 255.0
        newalpha_layer = newalpha_layer[:, :, np.newaxis]

        return newalpha_layer, newrgba_layer

    def delete_pic(self, dir_path):
        files = os.listdir(dir_path)
        # 删除文件夹中的所有 jpg 文件
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            try:
                if os.path.isfile(file_path) and file_name.lower().endswith('.jpg'):
                    os.remove(file_path)
                    # print(f"File '{file_path}' deleted successfully.")
                # else:
                #     pass
                    # print(f"Skipping non-JPG file: '{file_path}'")
            except Exception as e:
                print(f"Error deleting file '{file_path}': {str(e)}")

        print(f"All JPG files in folder '{dir_path}' deleted.")

    def show_qrcode(self, qr_path):
        self.is_showqr = True
        self.is_show_after = True
        # print("after:", qr_path)
        self.capture_window = ShowqrWindow(qr_path)
        def show_ui():
            self.capture_window.show()

        def on_ui_closed(is_closed):
            if is_closed:
                self.is_showqr = False
                self.is_show_after = False
                # print("uiclose")
        self.capture_window.closed.connect(on_ui_closed)  # 连接UI关闭信号
        show_ui()

    def qr_counter_update(self):
        self.qr_countdown_seconds -= 1
        # self.ui.label_2.setText(f"倒计时: {self.qr_close_counter_timer}秒")
        if self.qr_countdown_seconds == 0:
            self.qr_counter_timer.stop()
            # self.ui.label_2.setText("拍摄完成")
            self.qr_closeWindow()


    def clear_status(self):
        if self.show_real:
            self.ui.label_page4_counter.setText(" ")
        else:
            self.ui.label_3.setText(" ")
        self.capture_close_timer.stop()
    #关闭界面
    def qr_closeWindow(self):
        if self.capture_window is not None:
            self.capture_window.close()
            self.qr_counter_timer.stop()

    def set_bg_img_path(self, new_bg_img_path):
        self.bg_img_path = new_bg_img_path

    def crop_di(self, image,crop_width, crop_height ):
        # 获取图像的中心坐标
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        y = image.shape[0]
        # 计算裁剪区域的左上角和右下角坐标
        crop_x1 = max(center_x - crop_width // 2, 0)
        crop_x2 = min(center_x + crop_width // 2, image.shape[1])
        crop_y1 = max(y - crop_height, 0)
        crop_y2 = min(y, image.shape[0])

        # 利用切片操作裁剪图像
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped_image


    def crop_top(self, image,crop_width, crop_height,pos_x,pos_y):
        img_y = image.shape[0]
        img_x = image.shape[1]

        # 计算裁剪区域的左上角和右下角坐标
        x = min(np.abs(pos_x), (img_x - crop_width)//2 - 1)
        crop_x1 = int(max((image.shape[1] - crop_width) // 2, 0))
        if pos_x>0:
            start_x = crop_x1 + x
        else:
            start_x = crop_x1 - x
        crop_x2 = int(start_x + crop_width)
        crop_y1 = int(max(0, min(pos_y, int(img_y - crop_height-1))))
        crop_y2 = int(crop_y1 + crop_height)
        # 利用切片操作裁剪图像
        cropped_image = image[crop_y1:crop_y2, start_x:crop_x2]

        return cropped_image

    def crop_center(self, image, crop_width, crop_height):
        # 获取图像的中心坐标
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

        # 计算裁剪区域的左上角和右下角坐标
        crop_x1 = max(center_x - crop_width // 2, 0)
        crop_y1 = max(center_y - crop_height // 2, 0)
        crop_x2 = min(center_x + crop_width // 2, image.shape[1])
        crop_y2 = min(center_y + crop_height // 2, image.shape[0])

        # 利用切片操作裁剪图像
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped_image


    def select_camera(self, index):
        # 根据选择的相机切换
        if index == 0:  # 内置摄像头
            self.camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            self.camera.set(3, self.wCam)
            self.camera.set(4, self.hCam)
        elif index == 1:  # 外置摄像头
            # 注意：如果有外置摄像头，请修改设备索引（例如1，2，3...）
            self.camera = cv2.VideoCapture(1,cv2.CAP_DSHOW)
            self.camera.set(3, self.wCam)
            self.camera.set(4, self.hCam)

    def open_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '打开图片', './data/select_pic', 'Image Files (*.png *.jpg *.bmp)')
        if file_path:
            # 在这里调用 set_bg_img_path 方法，将新的图片路径传递给它
            new_bg_img_path = file_path
            self.set_bg_img_path(new_bg_img_path)
    def showContextMenu(self, pos):
        context_menu = QMenu(self)
        full_screen_action = context_menu.addAction("全屏")
        exit_full_screen_action = context_menu.addAction("退出全屏")

        full_screen_action.triggered.connect(self.toggleFullScreen)
        exit_full_screen_action.triggered.connect(self.exitFullScreen)

        context_menu.exec_(self.mapToGlobal(pos))

    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exitFullScreen(self):
        if self.isFullScreen():
            self.showNormal()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
