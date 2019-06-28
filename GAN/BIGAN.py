import numpy as np
import tensorflow as tf
import tensorview as tv


def image_encoder(latent_dim=100, image_shape=(28,28,1)):
    image = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Flatten()(image)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    encoder = tf.keras.layers.Dense(latent_dim)(x)
    encoder_net = tf.keras.Model(image, encoder)
    return encoder_net

def generator(latent_dim=100, image_shape=(28,28,1)):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(256)(noise)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(np.prod(image_shape), activation='tanh')(x)
    image = tf.keras.layers.Reshape(image_shape)(x)
    gnet = tf.keras.Model(noise, image)
    return gnet

def discriminator(latent_dim=100, image_shape=(28,28,1)):
    image_encoder = tf.keras.Input(shape=(latent_dim,))
    image = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Flatten()(image)
    x = tf.keras.layers.Concatenate()([image_encoder, x])
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    logit = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    dnet = tf.keras.Model([image_encoder, image], logit)
    return dnet

def train(batch_num=10000, batch_size=64, latent_dim=100, image_shape=(28,28,1)):
    dnet = discriminator(latent_dim, image_shape)
    dnet.compile(loss=['binary_crossentropy'],
                 optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 metrics=['accuracy'])

    noise = tf.keras.Input(shape=(latent_dim,))
    image = tf.keras.Input(shape=image_shape)
    gnet = generator(latent_dim, image_shape)
    encoder_net = image_encoder(latent_dim, image_shape)
    frozen = tf.keras.Model(dnet.inputs, dnet.outputs)
    frozen.trainable = False
    image_gen = gnet(noise)
    image_encode = encoder_net(image)
    logit_fake = frozen([noise, image_gen])
    logit_real = frozen([image_encode, image])
    bigan = tf.keras.Model([noise, image], [logit_fake, logit_real])
    bigan.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                  optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  metrics=['accuracy', 'accuracy'])

    (X_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    tv_plot = tv.train.PlotMetrics(columns=3, wait_num=50)
    for batch in range(batch_num):
        batch_image = X_train[np.random.choice(range(X_train.shape[0]), batch_size, False)]
        batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
        batch_image_gen = gnet.predict(batch_noise)
        batch_image_encode = encoder_net.predict(batch_image)

        d_loss_real = dnet.train_on_batch([batch_image_encode, batch_image], np.ones((batch_size, 1)))
        d_loss_fake = dnet.train_on_batch([batch_noise, batch_image_gen], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = bigan.train_on_batch([batch_noise,  batch_image], [np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        tv_plot.update({'D_loss': d_loss[0], 'D_binary_acc': d_loss[1],
                        'G_fake_loss': g_loss[1],  'G_real_loss': g_loss[2],
                        'G_fake_binary_acc': g_loss[3], 'G_real_binary_acc': g_loss[4]})
        tv_plot.draw()
    tv_plot.visual()
    tv_plot.visual(name='model_visual_gif', gif=True)
    return gnet


if __name__ == '__main__':
    gnet = train()
