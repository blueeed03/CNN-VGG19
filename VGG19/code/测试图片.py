# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:23:15 2023

@author: bluuue
"""
import tensorflow as tf
import numpy as np
import cv2
from yuchuli import ImageDataSet

# 加载已经训练好的模型
model = tf.keras.models.load_model(r"C:/Users/bluuue/Desktop/CNN/code/modelsaved_modelsaved_model")

# 加载并预处理输入图片
img_path = r'C:/Users/bluuue/Desktop/BurgerKing_8.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (64, 64))  # 根据需要调整大小


#如果输入为彩色三通道RGB图片则需进行灰度处理
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 如果模型期望灰度图像，则转换为灰度图像
img = np.stack((img,) * 3, axis=-1)


img = img / 255.0  # 将像素值归一化到 [0, 1]
img = np.expand_dims(img, axis=0)  # 添加批处理维度

# 进行推理
predictions = model(img)

# 根据需要处理预测结果
class_idx = np.argmax(predictions)

# 打印或使用预测结果
predict = [
'Benz',
'Buick',
'Citroen' ,
'FAW',
'Fukude',
'Honda',
'Hyundai' ,
'KIA',
'Lexus',
'Mazda',
'MG',
'Nissan',
'Toyota',
'Volkswagen' ,
'AoDi',
'BaoJun',
'BaoMa',
'BaoShiJie',
'BeiQiHuanSu',
'BeiQiWeiWang',
'BeiQiXinNengYuan',
'BenTeng',
'ChuanQi',
'DongFeng',
'DongNan',
'FengTianHuangGuan',
'JiLi',
'JiLiDiHao',
'JiLiQuanQiuYing',
'JiLiYingLun',
'JiPu',
'KaiRui',
'KaiDiLaKe',
'LiFan',
'QiRui',
'QiChen',
'SiBaLu',
'WeiLin',
'ZhongHua',
'ZhongTai']#种类标签

if class_idx >= len(predict):
    print("未能识别");
else:
    print(f"预测的类别索引为: {predict[class_idx]}")


