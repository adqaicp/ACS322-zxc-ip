import sys, os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, optimizers, metrics, layers, Sequential, Model, Input, losses
import time
from network import discriminator_network_cgan, generator_network_cgan
from load_datasets import DataLoader


def draw_loss(start_epochs, end_epochs, g_loss_list, d_loss_list):
    plt.plot(range(start_epochs, end_epochs), g_loss_list, label='g_loss', color='g')
    plt.plot(range(start_epochs, end_epochs), d_loss_list, label='d_loss', color='r')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(range(start_epochs, end_epochs + 1, save_step))
    plt.legend()
    plt.show()


def save_image(fake_image, fake_label=None, fake_num=16, save_to=''):
    plt.figure()
    _, h, w, c = fake_image.shape
    for i in range(fake_num):
        plt.subplot(4, 5, i + 1)
        if c == 3:  #
            plt.imshow(cv.cvtColor(fake_image[i], cv.COLOR_BGR2RGB))
        if c == 1:  #
            plt.imshow(np.squeeze(fake_image[i]), cmap='gray')
        if fake_label is not None:  #
            plt.title(fake_label[i])
        plt.axis('off')
    plt.tight_layout()  #
    print('Save figure to :%s' % save_to)
    plt.savefig(save_to)
    plt.close()


def creat_dir(path):
    os.makedirs(path, exist_ok=True)


# define parameters
label_start = 1
n_class = label_start + 20
batch_size = 4096
epochs = 50000
g_lr = 0.0002
d_lr = 0.0002
fake_num = 20
dataset_name = 'handwritten_characters'
dir_path = r'../datasets/%s' % dataset_name
h, w = 64, 64
save_step = 1000

# file folders
directory = './checkpoint/' + dataset_name + '_%s×%s' % (h, w)
image_dir = './result_image/' + dataset_name + '_%s×%s' % (h, w)
creat_dir(directory)
creat_dir(image_dir)
image_names = os.listdir(image_dir)

#
dataloader = DataLoader(dataset_name=dataset_name, norm_range=(-1, 1), img_res=(h, w))
# x_train, y_train, _, _ = dataloader.load_datasets(dir_path=r'datasets/%s' % dataset_name, test_size=0)  # 加载数据集
x_train, y_train = dataloader.load_datasets(dir_path, label_start=label_start)
x_train = tf.image.rgb_to_grayscale(x_train)  # 转为灰图
print(x_train.shape, x_train.numpy().min(), x_train.numpy().max(), y_train.shape)
_, h, w, c = x_train.shape
print(y_train)

#
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)
print(next(iter(train_db))[0].shape)

# network
generator = generator_network_cgan(h, w, c, n_class)
discriminator = discriminator_network_cgan(h, w, c, n_class)
generator.summary()  #
discriminator.summary()
g_optimizer = optimizers.Adam(learning_rate=g_lr, beta_1=0.9)  #
d_optimizer = optimizers.Adam(learning_rate=d_lr, beta_1=0.9)

# set rdm seed
tf.random.set_seed(0)
noise_test = tf.random.normal([fake_num, 100])
tf.random.set_seed(None)


#
@tf.function
def train(x, y):
    noise = tf.random.normal([x.shape[0], 100])  # (b,100)
    with tf.GradientTape(persistent=True) as tape:
        fake_image = generator([noise, y], training=True)  # (b,28,28,1)
        fake_out = discriminator([fake_image, y], training=True)  # (b,1) (b,10)
        real_out = discriminator([x, y], training=True)  # (b,1) (b,10)
        # loss
        g_loss = tf.reduce_mean(
            losses.binary_crossentropy(y_true=tf.ones_like(fake_out), y_pred=fake_out, from_logits=True))
        g_total_loss = g_loss
        # identify loss
        d1_loss = tf.reduce_mean(
            losses.binary_crossentropy(y_true=tf.zeros_like(fake_out), y_pred=fake_out, from_logits=True))
        d2_loss = tf.reduce_mean(
            losses.binary_crossentropy(y_true=tf.ones_like(real_out), y_pred=real_out, from_logits=True))
        d_total_loss = d1_loss + d2_loss

    #
    d_grads = tape.gradient(d_total_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    # d_accuracy.update_state(y_true=y_onehot, y_pred=real_label)
    g_grads = tape.gradient(g_total_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    # g_accuracy.update_state(y_true=noise_label_onehot, y_pred=fake_label)
    return [g_total_loss, d_total_loss]


#
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator,
                                 g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer)
checkpoint.restore(tf.train.latest_checkpoint(directory))  # 断点续训
manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=directory, max_to_keep=2,
                                     checkpoint_name=dataset_name)

max_num = max([int(image_name.split('.')[0][len(dataset_name):]) for image_name in image_names if
               image_name.endswith('png') or image_name.endswith('jpg')]) if image_names else 0  # save_path
t1 = time.time()
g_total_loss_list, d_total_loss_list = [], []
for epoch in range(epochs):  #
    g_list, d_list = [], []
    for x, y in train_db:
        g_loss, d_loss = train(x, y)
        g_list.append(g_loss)
        d_list.append(d_loss)
    g_total_loss = np.mean(g_list)
    d_total_loss = np.mean(d_list)
    # lost figure
    g_total_loss_list.append(g_total_loss)
    d_total_loss_list.append(d_total_loss)
    # 保存
    if save_step:
        if (epoch + 1) % save_step == 0 or epoch == 0:  #
            noise_test_label = np.array([11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                         13, 13, 13, 13, 13, 14, 14, 14, 14, 14])
            fake_image = generator([noise_test, noise_test_label], training=False)  # (15,h,w,c)
            fake_image = fake_image.numpy().reshape(-1, h, w, c) * 127.5 + 127.5  # 归一化回0-255为了保存图片
            fake_image = np.clip(fake_image, 0, 255).astype('uint8')  # (15,h,w,c)
            save_to = image_dir + '/' + dataset_name + '%d.png' % (epoch + 1 + max_num)
            save_image(fake_image, noise_test_label, fake_num, save_to)  #
            manager.save(checkpoint_number=epoch + 1 + max_num)  #
            print('epoch:%d/%d:' % (epoch + 1 + max_num, epochs),
                  'g_total_loss:%.6f' % np.array(g_total_loss),
                  'd_total_loss:%.6f' % np.array(d_total_loss), 'T:%.6f' % (time.time() - t1))
            t1 = time.time()
    if epoch + 1 + max_num >= epochs:
        print('Finished training!')
        break
manager.save(checkpoint_number=epochs + max_num)
try:
    draw_loss(max_num, epochs, g_total_loss_list, d_total_loss_list)  #
except:
    pass
