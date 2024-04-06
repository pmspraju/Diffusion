import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import callbacks
from readData import ReadWriteTFRecord, PrepareData, PrepareFlowerData
from utils import *

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

# butterfly dataset
prepareobj = PrepareData(parsed_image_dataset)
train_dataset, val_dataset, full_dataset = prepareobj.prepare_dataset()

# flower dataset
# split = "train[:80%]+validation[:80%]+test[:80%]"
# prepareobj = PrepareFlowerData(split)
# train_dataset = prepareobj.prepare_dataset()
# split = "train[80%:]+validation[80%:]+test[80%:]"
# val_dataset = prepareobj.prepare_dataset()

# Test the train dataset
# cnt = 0
# for bat in train_dataset:
#
#     for image in bat:
#         #plt.imshow(image[0].numpy())
#         #plt.savefig('image1.png')
#         #break
#         print(image.shape)
#         cnt += 1

from ddimModel import DiffusionModel
# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)

#[print(layer.name) for layer in model.layers]

model.compile(
    optimizer=tf.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=tf.losses.mean_absolute_error,
)

# save the best model based on the validation KID metric
checkpoint_path = os.path.join(checkpoint_path_butterfly, "diffusion_model.weights.h5")
#checkpoint_path = os.path.join(checkpoint_path_flower, "diffusion_model.weights.h5")
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_dataset)

#  Check if model exists to generate
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path, skip_mismatch=True)
    model.plot_images()
else:
    # run training and plot generated images periodically
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=[
            callbacks.LambdaCallback(on_epoch_end=model.plot_images),
            checkpoint_callback,
        ],
    )