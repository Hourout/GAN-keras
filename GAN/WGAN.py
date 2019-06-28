import numpy as np
import tensorflow as tf
import tensorview as tv
K = tf.keras.backend

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def generator(latent_dim=100, channels=1):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128 * 7 * 7, activation="relu")(noise)
    x = tf.keras.layers.Reshape((7, 7, 128))(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(128, 4, 1, "same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 4, 1, "same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channels, 4, 1, "same")(x)
    image = tf.keras.layers.Activation("tanh")(x)
    gnet = tf.keras.Model(noise, image)
    return gnet

def discriminator(image_shape=(28,28,1)):
    image = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Conv2D(16, 3, 2, "same")(image)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(32, 3, 2, "same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(64, 3, 2, "same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(128, 3, 1, "same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    logit = tf.keras.layers.Dense(1)(x)
    dnet = tf.keras.Model(image, logit)
    return dnet

def train(batch_num=10000, batch_size=128, latent_dim=100, image_shape=(28,28,1), opt_num=5, clip_value=0.01):
    dnet = discriminator(image_shape)
    dnet.compile(loss=wasserstein_loss, 
                 optimizer=tf.keras.optimizers.RMSprop(lr=0.00005),
                 metrics=['accuracy'])

    noise = tf.keras.Input(shape=(latent_dim,))
    gnet = generator(latent_dim)
    frozen = tf.keras.Model(dnet.inputs, dnet.outputs)
    frozen.trainable = False
    image = gnet(noise)
    logit = frozen(image)
    wgan = tf.keras.Model(noise, logit)
    wgan.compile(loss=wasserstein_loss,
                 optimizer=tf.keras.optimizers.RMSprop(lr=0.00005),
                 metrics=['accuracy'])

    (X_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    
    tv_plot = tv.train.PlotMetrics(columns=2, wait_num=5)
    for batch in range(batch_num):
        for _ in range(opt_num):
            batch_image = X_train[np.random.choice(range(X_train.shape[0]), batch_size, False)]
            batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
            batch_gen_image = gnet.predict(batch_noise)

            d_loss_real = dnet.train_on_batch(batch_image, -np.ones((batch_size, 1)))
            d_loss_fake = dnet.train_on_batch(batch_gen_image, np.ones((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            for l in dnet.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)
        g_loss = wgan.train_on_batch(batch_noise, -np.ones((batch_size, 1)))
        tv_plot.update({'D_loss': d_loss[0], 'D_binary_acc': d_loss[1],
                        'G_loss': g_loss[0], 'G_binary_acc': g_loss[1]})
        tv_plot.draw()
    tv_plot.visual()
    tv_plot.visual(name='model_visual_gif', gif=True)
    return gnet


if __name__ == '__main__':
    gnet = train()
