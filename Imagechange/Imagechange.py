# coding:utf-8
import os
from PIL import Image
import numpy as np


def resize(imgPath, savePath):
    files = os.listdir(imgPath)
    files.sort()
    print('****************')
    print('input :', imgPath)
    print('start...')
    for file in files:
        fileType = os.path.splitext(file)
        if fileType[1] == '.png':
            new_png = Image.open(imgPath + '/' + file)  # 打开图片
            # new_png = new_png.resize((20, 20),Image.ANTIALIAS) #改变图片大小
            matrix = 255 - np.asarray(new_png)  # 图像转矩阵 并反色
            new_png = Image.fromarray(matrix)  # 矩阵转图像
            new_png.save(savePath + '/' + file)  # 保存图片
    print('down!')
    print('****************')


if __name__ == '__main__':
    # 待处理图片地址
    dataPath = 'I:\\Individualproject\\HWDB1.0-binary\\00335'
    # 保存图片的地址
    savePath = 'I:\\Individualproject\\HWDB1.0-inverse\\00335'
    resize(dataPath, savePath)
