import numpy as np
import tensorflow as tf
import tensorview as tv


def generator(latent_dim=100, num_classes=10, image_shape=(28,28,1)):
    noise = tf.keras.Input(shape=(latent_dim,))
    label = tf.keras.Input(shape=(1,), dtype='int32')
    x = tf.keras.layers.Embedding(num_classes, latent_dim)(label)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Multiply()([noise, x])
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Dense(np.prod(image_shape), activation='tanh')(x)
    image = tf.keras.layers.Reshape(image_shape)(x)
    gnet = tf.keras.Model([noise, label], image)
    return gnet

def discriminator(image_shape=(28,28,1), num_classes=10):
    image = tf.keras.Input(shape=image_shape)
    label = tf.keras.Input(shape=(1,), dtype='int32')
    image_f = tf.keras.layers.Flatten()(image)
    x = tf.keras.layers.Embedding(num_classes, np.prod(image_shape))(label)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Multiply()([image_f, x])
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    logit = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    dnet = tf.keras.Model([image, label], logit)
    return dnet

def train(batch_num=10000, batch_size=64, latent_dim=100, num_classes=10, image_shape=(28,28,1)):
    dnet = discriminator(image_shape, num_classes)
    dnet.compile(loss=['binary_crossentropy'], 
                 optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 metrics=['accuracy'])

    noise = tf.keras.Input(shape=(latent_dim,))
    label = tf.keras.Input(shape=(1,), dtype='int32')
    gnet = generator(latent_dim, num_classes, image_shape)
    frozen = tf.keras.Model(dnet.inputs, dnet.outputs)
    frozen.trainable = False
    image = gnet([noise, label])
    logit = frozen([image, label])
    cgan = tf.keras.Model([noise, label], logit)
    cgan.compile(loss=['binary_crossentropy'],
                 optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 metrics=['accuracy'])

    (X_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)

    tv_plot = tv.train.PlotMetrics(columns=2, wait_num=50)
    for batch in range(batch_num):
        random = np.random.choice(range(X_train.shape[0]), batch_size, False)
        batch_image = X_train[random]
        batch_image_label = y_train[random]
        batch_noise = np.random.normal(0, 1, (batch_size, latent_dim))
        batch_noise_label = np.random.randint(0, num_classes, (batch_size, 1))

        batch_gen_image = gnet.predict([batch_noise, batch_image_label])

        d_loss_real = dnet.train_on_batch([batch_image, batch_image_label], np.ones((batch_size, 1)))
        d_loss_fake = dnet.train_on_batch([batch_gen_image, batch_image_label], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = cgan.train_on_batch([batch_noise, batch_noise_label], np.ones((batch_size, 1)))
        tv_plot.update({'D_loss': d_loss[0], 'D_binary_acc': d_loss[1],
                        'G_loss': g_loss[0], 'G_binary_acc': g_loss[1]})
        tv_plot.draw()
    tv_plot.visual()
    tv_plot.visual(name='model_visual_gif', gif=True)
    return gnet


if __name__ == '__main__':
    gnet = train()
