import sys, os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets
import sklearn
import re


def dir_name_sort(dir):  #
    dir_names = os.listdir(dir)
    new_dir_names = sorted(dir_names, key=lambda i: int(re.match(r'(\d+)', i).group()))
    return new_dir_names


class DataLoader():
    def __init__(self, dataset_name, norm_range=(0, 1), img_res=(64, 64)):

        self.dataset_name = dataset_name
        self.norm_range = norm_range
        self.img_res = img_res

    def img_normalize(self, image, norm_range=(0, 1)):

        image = np.array(image).astype('float32')
        image = (norm_range[1] - norm_range[0]) * image / 255. + norm_range[0]
        return image

    def load_datasets(self, dir_path, label_start=0):

        ImgTypeList = ['jpg', 'JPG', 'bmp', 'png', 'jpeg', 'rgb', 'tif']
        file_names = dir_name_sort(dir_path)
        x_total_list, y_total_list = [], []
        for i, file_name in enumerate(file_names):
            i += label_start
            file_name_path = os.path.join(dir_path, file_name)
            image_names = os.listdir(file_name_path)
            x_list, y_list = [], []
            for image_name in image_names:
                if image_name.split('.')[-1] in ImgTypeList:
                    image = cv.imread(os.path.join(file_name_path, image_name), flags=-1)
                    if image.shape[0] != self.img_res[1] or image.shape[1] != self.img_res[0]:
                        image = cv.resize(image,
                                          dsize=(self.img_res[1], self.img_res[0]))
                    if len(image.shape) == 2:
                        image = np.expand_dims(image, axis=-1)
                    x_list.append(image)
                    y_list.append(i)
            x, y = np.array(x_list), np.array(y_list)
            print(x.shape)
            x_total_list.append(x), y_total_list.append(y)
        x, y = np.concatenate(x_total_list, 0), np.concatenate(y_total_list, 0)
        print(x.shape)
        x = self.img_normalize(x, norm_range=(self.norm_range[0], self.norm_range[1]))
        return x, y


def main():
    height = 64
    width = 64
    # image_path = r'C:\Users\Administrator\Desktop\seeprettyface_wanghong'
    # from_dir_resize_image(image_path, height, width, save=True, to_save=r'C:\Users\Administrator\Desktop\star_images')
    dir_path = r'C:\Users\Administrator\Desktop\cgan\datasets\coil_data'
    dataloader = DataLoader(dataset_name='coil_data', norm_range=(-1, 1), img_res=(128, 128))
    # dataloader.save_all_images(dir_path)  # (10000, 64, 64, 3)
    x_train, y_train, x_test, y_test = dataloader.load_datasets(dir_path)
    print(x_train.shape, x_train.min(), x_train.max(), y_train.shape)


if __name__ == '__main__':
    main()
