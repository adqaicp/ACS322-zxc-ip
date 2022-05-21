import sys, os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from load_datasets import DataLoader
from network import discriminator_network_cgan, generator_network_cgan

# 定义参数（调参只需要改这里即可，其他地方不用改）
label_start = 1
n_class = label_start + 20
batch_size = 4096
epochs = 50000
g_lr = 0.0002
d_lr = 0.0002
fake_num = 20
dataset_name = 'handwritten_characters'
dir_path = r'../datasets/%s' % dataset_name
h, w, c = 40, 40, 1
save_step = 1

directory = './checkpoint/' + dataset_name + '_%s×%s' % (h, w)
image_dir = './result_image/' + dataset_name + '_%s×%s' % (h, w)

generator = generator_network_cgan(h, w, c, n_class)
discriminator = discriminator_network_cgan(h, w, c, n_class)
# generator.summary()
# discriminator.summary()
# 加载参数
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(directory))
# 预测
noise_test = tf.random.normal([fake_num, 100])

noise_test_label = np.array([12, 12, 12, 12, 11, 11, 11, 11, 12, 12,
                             13, 13, 13, 13, 13, 14, 14, 14, 14, 14])
# noise_test_label = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
fake_image = generator([noise_test, noise_test_label], training=False)  # (15,h,w,c)
fake_image = fake_image.numpy().reshape(-1, h, w, c) * 127.5 + 127.5  # (15,h,w,c) 归一化回0-255为了展示图片

plt.figure()
_, h, w, c = fake_image.shape
for i in range(fake_num):
    plt.subplot(4, 5, i + 1)
    if c == 3:
        plt.imshow(cv.cvtColor(fake_image[i], cv.COLOR_BGR2RGB))
    if c == 1:
        plt.imshow(np.squeeze(fake_image[i]), cmap='gray')
    plt.title(noise_test_label[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
