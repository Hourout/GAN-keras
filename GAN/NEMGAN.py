import numpy as np
import tensorflow as tf
import tensorview as tv
K = tf.keras.backend


def AMSoftmax(inputs, units, s, m):
    dense = tf.keras.layers.Dense(units)(inputs)
    psi = tf.keras.layers.Lambda(lambda x:K.exp(s*(x-m)))(dense)
    costheta = tf.keras.layers.Lambda(lambda x:K.exp(s*x))(dense)
    costheta_sum = tf.keras.layers.Lambda(lambda x:K.sum(x, axis=-1, keepdims=True))(costheta)
    sub = tf.keras.layers.Subtract()([psi, costheta])
    add = tf.keras.layers.Add()([sub, costheta_sum])
    output = tf.keras.layers.Lambda(lambda x:x[0]/x[1])([psi, add])
    return output
        
def decoder(latent_dim=100, image_shape=(28,28,1), dropout=0.4, depth=32):
    image = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Conv2D(depth, 5, strides=2, padding='same')(image)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(depth*2, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(depth*4, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(depth*8, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(depth*16, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x2 = AMSoftmax(x1, 10, 30, 0.4)
    deco = tf.keras.Model(image, [x1, x2])
    return deco

def discriminator(image_shape=(28,28,1), dropout=0.4, depth=32*2):
    image = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Conv2D(depth*1, 5, strides=2, padding='same')(image)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(depth*2, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(depth*4, 5, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(depth*8, 5, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Flatten()(x)
    logit = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    dnet = tf.keras.Model(image, logit)
    return dnet

def generator(latent_dim=100, dropout=0.3, depth=32*4, dim=7):
    noise = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(1024)(noise)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(dim*dim*depth)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Reshape((dim, dim, depth))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2DTranspose(int(depth/2), 5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2DTranspose(int(depth/4), 5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(int(depth/8),5 , padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(int(depth/16), 5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(int(depth/32),5, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    image = tf.keras.layers.Conv2DTranspose(1, 5, activation='sigmoid', padding='same')(x)
    gnet = tf.keras.Model(noise, image)
    return gnet

def amsoftmax_loss(y_true, y_pred):
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss

def train(batch_num=10000, batch_size=64, latent_dim=100, image_shape=(28,28,1), num_classes=10):
    dnet = discriminator()
    dnet.compile(loss='binary_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(lr=0.0002),
                 metrics=['acc'])

    noise = tf.keras.Input(shape=(latent_dim,))
    gnet = generator()
    frozen = tf.keras.Model(dnet.inputs, dnet.outputs)
    frozen.trainable = False
    image = gnet(noise)
    logit = frozen(image)
    nemgan = tf.keras.Model(noise, logit)
    nemgan.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(lr=0.0002),
                   metrics=['acc'])

    decoder_net = decoder()
    image_decode, image_label = decoder_net(image)
    decoder_net = tf.keras.Model(noise, [image_decode, image_label])
    decoder_net.compile(loss=['mse', amsoftmax_loss],
                        loss_weights=[100.0, 1.0],
                        optimizer=tf.keras.optimizers.Adam(lr=0.00001, decay=6e-8),
                        metrics=['mae', 'acc'])

    (X_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    tv_plot = tv.train.PlotMetrics(columns=2, wait_num=50)
    for batch in range(batch_num):
        batch_image = X_train[np.random.choice(range(X_train.shape[0]), batch_size, False)]
        batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
        batch_noise_label = tf.keras.utils.to_categorical(batch_noise[:,:10].argmax(1), num_classes)
        batch_image_gen = gnet.predict(batch_noise)

        d_loss_real = dnet.train_on_batch(batch_image, np.ones((batch_size, 1)))
        d_loss_fake = dnet.train_on_batch(batch_image_gen, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = nemgan.train_on_batch(batch_noise, np.ones((batch_size, 1)))
        decoder_loss = decoder_net.train_on_batch(batch_noise, [batch_noise, batch_noise_label])
        tv_plot.update({'D_loss': d_loss[0], 'D_binary_acc': d_loss[1],
                        'G_loss': g_loss[0],  'G_binary_acc': g_loss[1],
                        'AE_MSE': decoder_loss[1], 'AE_MAE': decoder_loss[2],
                        'AE_amsoftmax_loss':decoder_loss[3], 'AE_acc': decoder_loss[4]})
        tv_plot.draw()
    tv_plot.visual()
    tv_plot.visual(name='model_visual_gif', gif=True)
    return gnet


if __name__ == '__main__':
    gnet = train()
