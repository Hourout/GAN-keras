import scipy
import numpy as np
import tensorflow as tf
import tensorview as tv


def generator(latent_dim=100, image_shape=(28,28,1)):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(256)(noise)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

    image1 = tf.keras.layers.Dense(1024)(x)
    image1 = tf.keras.layers.LeakyReLU(alpha=0.2)(image1)
    image1 = tf.keras.layers.BatchNormalization(momentum=0.8)(image1)
    image1 = tf.keras.layers.Dense(np.prod(image_shape), activation='tanh')(image1)
    image1 = tf.keras.layers.Reshape(image_shape)(image1)

    image2 = tf.keras.layers.Dense(1024)(x)
    image2 = tf.keras.layers.LeakyReLU(alpha=0.2)(image2)
    image2 = tf.keras.layers.BatchNormalization(momentum=0.8)(image2)
    image2 = tf.keras.layers.Dense(np.prod(image_shape), activation='tanh')(image2)
    image2 = tf.keras.layers.Reshape(image_shape)(image2)

    gnet1 = tf.keras.Model(noise, image1)
    gnet2 = tf.keras.Model(noise, image2)
    return gnet1, gnet2

def discriminator(image_shape=(28,28,1)):
    image1 = tf.keras.Input(shape=image_shape)
    image2 = tf.keras.Input(shape=image_shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    logit1 = tf.keras.layers.Dense(1, activation='sigmoid')(model(image1))
    logit2 = tf.keras.layers.Dense(1, activation='sigmoid')(model(image2))

    dnet1 = tf.keras.Model(image1, logit1)
    dnet2 = tf.keras.Model(image2, logit2)
    return dnet1, dnet2

def train(batch_num=10000, batch_size=64, latent_dim=100, image_shape=(28,28,1)):
    dnet1, dnet2 = discriminator(image_shape)
    dnet1.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  metrics=['accuracy'])
    dnet2.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  metrics=['accuracy'])

    noise = tf.keras.Input(shape=(latent_dim,))
    gnet1, gnet2 = generator(latent_dim, image_shape)
    image1 = gnet1(noise)
    image2 = gnet2(noise)
    frozen1 = tf.keras.Model(dnet1.inputs, dnet1.outputs)
    frozen1.trainable = False
    frozen2 = tf.keras.Model(dnet2.inputs, dnet2.outputs)
    frozen2.trainable = False
    logit1 = frozen1(image1)
    logit2 = frozen2(image2)
    cogan = tf.keras.Model(noise, [logit1, logit2])
    cogan.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                  optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  metrics=['accuracy', 'accuracy'])

    (X_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    x1 = X_train[:int(X_train.shape[0]/2)]
    x2 = X_train[int(X_train.shape[0]/2):]
    x2 = scipy.ndimage.interpolation.rotate(x2, 90, axes=(1, 2))

    tv_plot = tv.train.PlotMetrics(columns=2, wait_num=50)
    for batch in range(batch_num):
        random = np.random.choice(range(x1.shape[0]), batch_size, False)
        batch_image1, batch_image2 = x1[random], x2[random]
        batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
        batch_gen_image1 = gnet1.predict(batch_noise)
        batch_gen_image2 = gnet2.predict(batch_noise)

        d1_loss_real = dnet1.train_on_batch(batch_image1, np.ones((batch_size, 1)))
        d2_loss_real = dnet2.train_on_batch(batch_image2, np.ones((batch_size, 1)))
        d1_loss_fake = dnet1.train_on_batch(batch_gen_image1, np.zeros((batch_size, 1)))
        d2_loss_fake = dnet2.train_on_batch(batch_gen_image2, np.zeros((batch_size, 1)))
        d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
        d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

        g_loss = cogan.train_on_batch(batch_noise, [np.ones((batch_size, 1)), np.ones((batch_size, 1))])
        tv_plot.update({'D1_loss': d1_loss[0], 'D1_binary_acc': d1_loss[1],
                        'D2_loss': d2_loss[0], 'D2_binary_acc': d2_loss[1],
                        'G1_loss': g_loss[1], 'G1_binary_acc': g_loss[3],
                        'G2_loss': g_loss[2], 'G2_binary_acc': g_loss[4]})
        tv_plot.draw()
    tv_plot.visual()
    tv_plot.visual(name='model_visual_gif', gif=True)
    return gnet1, gnet2


if __name__ == '__main__':
    gnet1, gnet2 = train()
