import numpy as np
import tensorflow as tf
import tensorview as tv


def generator(latent_dim=100):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128 * 7 * 7, activation="relu")(noise)
    x = tf.keras.layers.Reshape((7, 7, 128))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Conv2D(1, kernel_size=3, padding="same")(x)
    image = tf.keras.layers.Activation("tanh")(x)
    gnet = tf.keras.Model(noise, image)
    return gnet

def discriminator(num_classes=10, image_shape=(28,28,1)):
    image = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(image)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    valid = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    label = tf.keras.layers.Dense(num_classes+1, activation="softmax")(x)
    dnet = tf.keras.Model(image, [valid, label])
    return dnet

def train(batch_num=10000, batch_size=64, latent_dim=100, num_classes=10, image_shape=(28,28,1)):
    dnet= discriminator(num_classes, image_shape)
    dnet.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                 optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 metrics=['accuracy'])

    noise = tf.keras.Input(shape=(latent_dim,))
    gnet = generator(latent_dim)
    frozen = tf.keras.Model(dnet.inputs, dnet.outputs)
    frozen.trainable = False
    image = gnet(noise)
    valid, _ = frozen(image)
    sgan = tf.keras.Model(noise, valid)
    sgan.compile(loss=['binary_crossentropy'],
                 optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 metrics=['accuracy'])

    (X_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    y_train = tf.keras.utils.to_categorical(y_train.reshape(-1, 1), num_classes+1)
    w1 = {0: 1, 1: 1}
    w2 = {i: num_classes /(batch_size//2) for i in range(num_classes)}
    w2[num_classes] = 1 /(batch_size//2)
    
    tv_plot = tv.train.PlotMetrics(columns=2, wait_num=50)
    for batch in range(batch_num):
        random = np.random.choice(range(X_train.shape[0]), batch_size, False)
        batch_image = X_train[random]
        batch_label = y_train[random]
        batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
        batch_noise_label = tf.keras.utils.to_categorical(np.full((batch_size, 1), num_classes), num_classes+1)
        batch_image_gen = gnet.predict(batch_noise)

        d_loss_real = dnet.train_on_batch(batch_image, [np.ones((batch_size, 1)), batch_label], class_weight=[w1, w2])
        d_loss_fake = dnet.train_on_batch(batch_image_gen, [np.zeros((batch_size, 1)), batch_noise_label], class_weight=[w1, w2])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = sgan.train_on_batch(batch_noise,  np.ones((batch_size, 1)))
        tv_plot.update({'D_binary_loss': d_loss[1], 'D_binary_acc': d_loss[3],
                        'D_categorical_loss': d_loss[2], 'D_categorical_acc': d_loss[4],
                        'G_loss': g_loss[0],  'G_acc': g_loss[1]})
        tv_plot.draw()
    tv_plot.visual()
    tv_plot.visual(name='model_visual_gif', gif=True)
    return gnet


if __name__ == '__main__':
    gnet = train()
