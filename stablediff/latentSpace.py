import keras_cv
import keras
import matplotlib.pyplot as plt
from keras import backend as ops
import numpy as np
import math
from PIL import Image
import tensorflow as tf

# Enable mixed precision
# (only do this if you have a recent NVIDIA GPU)
keras.mixed_precision.set_global_policy("mixed_float16")

# Instantiate the Stable Diffusion model
model = keras_cv.models.StableDiffusion(jit_compile=True)

interpolation_steps = 150
batch_size = 3
batches = interpolation_steps // batch_size
seed = 12345
noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

prompt_1 = "A watercolor painting of the beautiful Hermione Granger smiling"
prompt_2 = "A watercolor painting of a beautiful dove flying in an open blue sky"

encoding_1 = tf.squeeze(model.encode_text(prompt_1))
encoding_2 = tf.squeeze(model.encode_text(prompt_2))

interpolated_encodings = tf.linspace(encoding_1, encoding_2, interpolation_steps)
batched_encodings = tf.split(interpolated_encodings, batches)

images = []
for batch in range(batches):
    images += [
        Image.fromarray(img)
        for img in model.generate_image(
            batched_encodings[batch],
            batch_size=batch_size,
            num_steps=25,
            diffusion_noise=noise,
        )
    ]

def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )

export_as_gif("jc-and-dove-150.gif", images, rubber_band=True)

prompt = "A watercolor painting of a majestic harpy eagle flying high in an open windy blue sky"
encoding = tf.squeeze(model.encode_text(prompt))
walk_steps = 150
batch_size = 3
batches = walk_steps // batch_size

walk_noise_x = tf.random.normal(noise.shape, dtype="float64")
walk_noise_y = tf.random.normal(noise.shape, dtype="float64")

walk_scale_x = ops.cos(tf.linspace(0, 2, walk_steps) * math.pi)
walk_scale_y = ops.sin(tf.linspace(0, 2, walk_steps) * math.pi)
noise_x = tf.tensordot(walk_scale_x, walk_noise_x, axes=0)
noise_y = tf.tensordot(walk_scale_y, walk_noise_y, axes=0)
noise = tf.add(noise_x, noise_y)
batched_noise = tf.split(noise, batches)

# images = []
# for batch in range(batches):
#     images += [
#         Image.fromarray(img)
#         for img in model.generate_image(
#             encoding,
#             batch_size=batch_size,
#             num_steps=25,
#             diffusion_noise=batched_noise[batch],
#         )
#     ]
#
# export_as_gif("eagle.gif", images)
