import pandas as pd
import os
from tensorflow.keras import callbacks
from readData import ReadWriteTFRecord, PrepareData
from utils import *
import tensorflow_datasets as tfds

os.environ["KERAS_BACKEND"] = "tensorflow"

datapath = r'/home/nachiketa/Documents/Workspaces/data/Stablediffusion/smithsonian_butterflies_subset/data'

# print(df.columns)
# image = tf.image.decode_image(df['image'][1].get('bytes'))
# plt.imshow(image.numpy())
# plt.savefig('image.png')

readobj = ReadWriteTFRecord(datapath)
filename = 'images.tfrecords'

# Write the tfrecord file from parquet
if not os.path.exists(os.path.join(datapath, filename)):
    filename = 'train-00000-of-00001.parquet'
    path = os.path.join(datapath, filename)
    df = pd.read_parquet(path)
    readobj.writetfrecord(df, filename)

# Read the tfrecord file
parsed_image_dataset = readobj.readtfrecord(filename)
# Test the tfrecord file
# for image_features in parsed_image_dataset.take(1):
#     image_raw = image_features['image_raw'].numpy()
#     image = tf.image.decode_png(image_raw)
#
#     image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
#     image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
#
#     plt.imshow(image.numpy())
#     plt.savefig('image1.png')

# flower dataset
(ds,) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)
prepareobj = PrepareData(ds)
train_dataset = prepareobj.create_dataset()
#Test the train dataset
# cnt = 0
# for bat in train_dataset:
#
#     for image in bat:
#         plt.imshow(image[0].numpy())
#         plt.savefig('image1.png')
#         break
#         #print(image.shape)
#         cnt += 1

from unet import build_model

# Build the unet model
network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)

ema_network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)
ema_network.set_weights(network.get_weights())  # Initially the weights are the same

from gaussianDiffusion import GaussianDiffusion

# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

from ddpmodel import DiffusionModel

# Get the model
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

# Compile the model
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

from tensorflow.keras import callbacks

checkpoint_path = os.path.join(checkpoint_path_flower, "ddpm_model.weights.h5")
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

# Train the model
model.fit(
    train_dataset,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
               checkpoint_callback,
               ],
)




