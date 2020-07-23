import numpy as np
from tensorflow.keras import Sequential, layers
from matplotlib.image import imsave

latent_dim = 100  # latent dimension for generator
impath = "imgs/"  # path to save images

print("====Creating Generator====")


def generator(latent_dim):
    model = Sequential()
    # 64x64x4 -> 16x16x4 (Dense layer for lower res image)
    n_nodes = 8*8*1024  # 1024 for multiple versions
    model.add(layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.5))
    model.add(layers.Reshape((8, 8, 1024)))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'))  # noqa: E501
    model.add(layers.BatchNormalization(momentum=0.5))
    model.add(layers.ReLU())
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))  # noqa: E501
    model.add(layers.BatchNormalization(momentum=0.5))
    model.add(layers.ReLU())
    # upsample to 64x64
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))  # noqa: E501
    model.add(layers.BatchNormalization(momentum=0.5))
    model.add(layers.ReLU())
    # output
    model.add(layers.Conv2D(4, (3, 3), activation='tanh', padding='same'))
    return model


g = generator(latent_dim)
g.load_weights('generator_model_200_epochs.h5')
print("Generator Weights Loaded")


def generate_fake_samples(generator, latent_dim, n_samples):
    noise = np.random.randn(latent_dim * n_samples)
    noise = noise.reshape(n_samples, latent_dim)
    X = generator.predict(noise)
    return X


print("Generating Images")
generated_imgs = generate_fake_samples(g, latent_dim, 10)

for i, img in enumerate(generated_imgs):
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = img.clip(min=0, max=1)
    imsave(impath + 'img{}.png'.format(i), img)

print("==={0} Images saved in {1}===".format(len(generated_imgs), impath))
