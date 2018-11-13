import numpy as np
import livelossplot
import tensorflow as tf


def generator(latent_dim=100, channels=3):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128 * 6 * 6, activation='relu')(noise)
    x = tf.keras.layers.Reshape((6, 6, 128))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(channels, kernel_size=3, padding='same')(x)
    image = tf.keras.layers.Activation('tanh')(x)
    gnet = tf.keras.Model(noise, image)
    return gnet

def discriminator(image_shape=(96,96,3)):
    image = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(image)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    logit = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dnet = tf.keras.Model(image, logit)
    return dnet

def train(batch_num=10000, batch_size=64, latent_dim=100, image_shape=(96,96,3)):
    tf.logging.set_verbosity(tf.logging.ERROR)
    dnet = discriminator(image_shape)
    dnet.compile(loss='binary_crossentropy', 
                 optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 metrics=['accuracy'])

    noise = tf.keras.Input(shape=(latent_dim,))
    gnet = generator(latent_dim, channels=image_shape[2])
    dnet.trainable = False
    image = gnet(noise)
    logit = dnet(image)
    dcgan = tf.keras.Model(noise, logit)
    dcgan.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  metrics=['accuracy'])

    dataset = (tf.data.Dataset.list_files('./faces/*.jpg')
        .map(lambda x: (tf.cast(tf.image.decode_jpeg(tf.read_file(x)), tf.float64)/127.5-1,
                        tf.ones(1, tf.int32)))
        .batch(batch_size)
        .repeat())
    iterator = dataset.make_one_shot_iterator()
    (batch_image, target) = iterator.get_next()
    
    liveplot = livelossplot.PlotLosses(max_cols=2)
    for batch in range(batch_num):
        d_batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
        batch_gen_image = gnet.predict(d_batch_noise)

        d_loss_real = dnet.train_on_batch(batch_image, target)
        d_loss_fake = dnet.train_on_batch(batch_gen_image, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = dcgan.train_on_batch(d_batch_noise, np.ones((batch_size, 1)))
        if batch%1==0:
            liveplot.update({'D_loss': d_loss[0], 'D_binary_acc': d_loss[1],
                             'G_loss': g_loss[0], 'G_binary_acc': g_loss[1]})
            liveplot.draw()
    return gnet


if __name__ == '__main__':
    gnet = train()
