from tensorflow.keras import Input, layers, Model
import numpy as np


def generator_network_cgan(h, w, c, n_class, noise_dim=100):
    input_label = Input(shape=(1,), dtype='int32')
    label = layers.Embedding(input_dim=n_class, output_dim=noise_dim)(input_label)
    label = layers.Flatten()(label)
    input_image = Input(shape=(100,))  #
    total_input = layers.Multiply()([input_image, label])
    x = layers.Dense(256)(total_input)  # (b,256)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(512)(x)  # (b,512)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(1024)(x)  # (b,1024)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(np.prod([h, w, c]), activation='tanh')(x)  # (b,784)
    x = layers.Reshape((h, w, c))(x)  # (b,28,28,1)

    generator = Model([input_image, input_label], x)
    return generator


def discriminator_network_cgan(h, w, c, n_class):  #
    input_label = Input(shape=(1,), dtype='int32')  # (b,1)
    label = layers.Embedding(input_dim=n_class, output_dim=np.prod([h, w, c]))(input_label)  # (b,1,h*w*c)

    label = layers.Flatten()(label)  # (b,h*w*c)
    input_image = Input(shape=(h, w, c))  # (b,h,w,c)
    image = layers.Flatten()(input_image)  # (b,h*w*c)
    total_input = layers.Multiply()([image, label])  # (b,h*w*c)

    x = layers.Dense(512)(total_input)  # (b,512)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256)(x)  # (b,256)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    validity = layers.Dense(1, activation=None)(x)  # (b,1)

    discriminator = Model([input_image, input_label], validity)
    return discriminator


def main():
    image = np.zeros([100, 64, 64, 3])
    b, h, w, c = image.shape
    n_class = 10
    # generator_network(h, w, c, n_class=n_class)
    discriminator = discriminator_network_cgan(h, w, c, n_class)


if __name__ == '__main__':
    main()
