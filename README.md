

# 实时背景替换以及人像抠图程序


<!-- PROJECT SHIELDS -->
![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![python version](https://img.shields.io/badge/paddlepaddle_gpu-2.3.2+-red.svg)
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->

[//]: # (<br />)

[//]: # ()
[//]: # (<p align="center">)

[//]: # (  <a href="https://github.com/shaojintian/Best_README_template/">)

[//]: # (    <img src="images/logo.png" alt="Logo" width="80" height="80">)

[//]: # (  </a>)

[//]: # ()
[//]: # (  <h3 align="center">"完美的"README模板</h3>)

[//]: # (  <p align="center">)

[//]: # (    一个"完美的"README模板去快速开始你的项目！)

[//]: # (    <br />)

[//]: # (    <a href="https://github.com/shaojintian/Best_README_template"><strong>探索本项目的文档 »</strong></a>)

[//]: # (    <br />)

[//]: # (    <br />)

[//]: # (    <a href="https://github.com/shaojintian/Best_README_template">查看Demo</a>)

[//]: # (    ·)

[//]: # (    <a href="https://github.com/shaojintian/Best_README_template/issues">报告Bug</a>)

[//]: # (    ·)

[//]: # (    <a href="https://github.com/shaojintian/Best_README_template/issues">提出新特性</a>)

[//]: # (  </p>)

[//]: # (</p>)

## 展示
https://github.com/7yzx/bg_remover/assets/86868727/a512cd32-3268-4caf-8762-7b7e7a482cd0


## 目录
- [功能介绍](#功能介绍)
- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [参考](#参考)
- [文件目录说明](#文件目录说明)
- [部署](#部署)
- [版本控制](#版本控制)
- [说明](#说明)


### 功能介绍
- 程序可打包，在一体机上运行
- 实时，且支持n多背景图片；采用抠图处理效果更好；可对图片进行多种处理（加水印，logo等）
- 人像位置，大小，摄像头视角自定义
- 可使用手势进行控制拍照计时，（鼠标对CPU要求高）

### 上手指南

###### 开发前的配置要求
主要参考了PaddleSeg中的源码来实现，在两者的基础上做了结合。配置基本没有问题，使用的release版本2.9  paddlegpu为2.3.2
1. [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/contrib/PP-HumanSeg/README_cn.md)
2. [Paddle Matting](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/Matting)
###### 参考
```bash
pip -r requirements.txt
```
### 文件目录说明
```
filetree 
├── README.md
├── LICENSE.txt
├── main_demo2.py#主程序
├── util.py#工具文件
├── /paddleseg1/
├── /paddleseg/
├── /Matting/
├── /data/ #图片存放
├── /config/ # 参数存放

```

### 部署

暂无



### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。


### 说明
本仓库仅作展示，本项目有exe程序打包和UI暂不公开。可联系拓展QQ：2804006356

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt)


<!-- links -->
[your-project-path]: 7yzx/bg_remover
[forks-shield]: https://img.shields.io/github/forks/7yzx/bg_remover.svg?style=flat-square
[forks-url]: https://github.com/7yzx/bg_remover/network/members
[stars-shield]: https://img.shields.io/github/stars/7yzx/bg_remover.svg?style=flat-square
[stars-url]: https://github.com/7yzx/bg_remover/stargazers
[issues-shield]: https://img.shields.io/github/issues/7yzx/bg_remover.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/7yzx/bg_remover.svg
[license-shield]: https://img.shields.io/github/license/7yzx/bg_remover.svg?style=flat-square
[license-url]: https://github.com/7yzx/bg_remover/blob/master/LICENSE.txt





