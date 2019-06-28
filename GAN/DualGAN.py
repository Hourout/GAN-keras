import scipy
import numpy as np
import tensorflow as tf
import tensorview as tv
K = tf.keras.backend

def generator(latent_dim=784):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(256)(noise)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(latent_dim, activation='tanh')(x)
    gnet = tf.keras.Model(noise, x)
    return gnet

def discriminator(latent_dim=784):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(512)(noise)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    logit = tf.keras.layers.Dense(1)(x)
    dnet = tf.keras.Model(noise, logit)
    return dnet

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def train(batch_num=1000, latent_dim=784, batch_size=128, n_critic=4, clip_value=0.01):
    dnet_a = discriminator(latent_dim)
    dnet_a.compile(loss=wasserstein_loss,
                   optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                   metrics=['accuracy'])
    dnet_b = discriminator(latent_dim)
    dnet_b.compile(loss=wasserstein_loss,
                   optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                   metrics=['accuracy'])
    image_a = tf.keras.Input(shape=(latent_dim,))
    image_b = tf.keras.Input(shape=(latent_dim,))
    gnet_a = generator(latent_dim)
    gnet_b = generator(latent_dim)
    image_gen_a = gnet_a(image_a)
    image_gen_b = gnet_b(image_b)
    frozen_a = tf.keras.Model(dnet_a.inputs, dnet_a.outputs)
    frozen_a.trainable = False
    frozen_b = tf.keras.Model(dnet_b.inputs, dnet_b.outputs)
    frozen_b.trainable = False
    logit_a = frozen_a(image_gen_b)
    logit_b = frozen_b(image_gen_a)
    recov_b = gnet_a(image_gen_b)
    recov_a = gnet_b(image_gen_a)
    dualgan = tf.keras.Model([image_a, image_b], [logit_a, logit_b, recov_a, recov_b])
    dualgan.compile(loss=[wasserstein_loss, wasserstein_loss, 'mae', 'mae'],
                    optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                    loss_weights=[1, 1, 100, 100])
    
    (X_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_a = X_train[:int(X_train.shape[0]/2)].reshape([-1, latent_dim])
    X_b = scipy.ndimage.interpolation.rotate(X_train[int(X_train.shape[0]/2):], 90, axes=(1, 2)).reshape([-1, latent_dim])

    tv_plot = tv.train.PlotMetrics(columns=2, wait_num=5)
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))
    for batch in range(batch_num):
        for _ in range(n_critic):
            batch_image_a = X_a[np.random.choice(range(X_a.shape[0]), batch_size, False)]
            batch_image_b = X_b[np.random.choice(range(X_b.shape[0]), batch_size, False)]
            batch_image_gen_b = gnet_a.predict(batch_image_a)
            batch_image_gen_a = gnet_b.predict(batch_image_b)
            d_a_loss_real = dnet_a.train_on_batch(batch_image_a, valid)
            d_a_loss_fake = dnet_a.train_on_batch(batch_image_gen_a, fake)
            d_b_loss_real = dnet_b.train_on_batch(batch_image_b, valid)
            d_b_loss_fake = dnet_b.train_on_batch(batch_image_gen_b, fake)
            d_a_loss = 0.5 * np.add(d_a_loss_real, d_a_loss_fake)
            d_b_loss = 0.5 * np.add(d_b_loss_real, d_b_loss_fake)
            for d in [dnet_a, dnet_b]:
                for l in d.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)
        g_loss = dualgan.train_on_batch([batch_image_a, batch_image_b], [valid, valid, batch_image_a, batch_image_b])
        tv_plot.update({'D_a_loss': d_a_loss[0], 'D_a_binary_acc': d_a_loss[1],
                        'D_b_loss': d_b_loss[0], 'D_b_binary_acc': d_b_loss[1],
                        'G_a_loss': g_loss[1],  'G_b_loss': g_loss[2]})
        tv_plot.draw()
    tv_plot.visual()
    tv_plot.visual(name='model_visual_gif', gif=True)
    return gnet_a, gnet_b


if __name__ == '__main__':
    gnet_a, gnet_b = train()
