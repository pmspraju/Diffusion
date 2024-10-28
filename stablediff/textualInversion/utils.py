EPOCHS = 50

MAX_PROMPT_LENGTH   = 77
IMAGE_HEIGHT        = 512
IMAGE_WIDTH         = 512

MAX_PROMPT_LENGTH   = 77
PLACEHOLDER_TOKEN   = "funcatok" #<my-funny-cat-token>

URLS_SINGLE         = [
        "https://i.imgur.com/VIedH1X.jpg",
        "https://i.imgur.com/eBw13hE.png",
        "https://i.imgur.com/oJ3rSg7.png",
        "https://i.imgur.com/5mCL6Df.jpg",
        "https://i.imgur.com/4Q6WWyI.jpg",
    ]

CHECK_POINT_PATH = r'/home/nachiketa/Documents/Workspaces/checkpoints//textualinversion/model/'

SAVE_IMAGE_PATH = r'/home/nachiketa/Documents/Workspaces/checkpoints/textualinversion/images/'

LOCAL_SINGLE = r'/home/nachiketa/Documents/Workspaces/data/Stablediffusion/textualinversion/single/'

PROMPTS_SINGLE      = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

URLS_GROUP          = [
        "https://i.imgur.com/yVmZ2Qa.jpg",
        "https://i.imgur.com/JbyFbZJ.jpg",
        "https://i.imgur.com/CCubd3q.jpg",
    ]

LOCAL_GROUP = r'/home/nachiketa/Documents/Workspaces/data/Stablediffusion/textualinversion/group/'

PROMPTS_GROUP       = [
        "a photo of a group of {}",
        "a rendering of a group of {}",
        "a cropped photo of the group of {}",
        "the photo of a group of {}",
        "a photo of a clean group of {}",
        "a photo of my group of {}",
        "a photo of a cool group of {}",
        "a close-up photo of a group of {}",
        "a bright photo of the group of {}",
        "a cropped photo of a group of {}",
        "a photo of the group of {}",
        "a good photo of the group of {}",
        "a photo of one group of {}",
        "a close-up photo of the group of {}",
        "a rendition of the group of {}",
        "a photo of the clean group of {}",
        "a rendition of a group of {}",
        "a photo of a nice group of {}",
        "a good photo of a group of {}",
        "a photo of the nice group of {}",
        "a photo of the small group of {}",
        "a photo of the weird group of {}",
        "a photo of the large group of {}",
        "a photo of a cool group of {}",
        "a photo of a small group of {}",
    ]

## Utility methods

import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
def traverse_layers(layer):
    if hasattr(layer, "layers"):
        for layer in layer.layers:
            yield layer
    if hasattr(layer, "token_embedding"):
        yield layer.token_embedding
    if hasattr(layer, "position_embedding"):
        yield layer.position_embedding

# sample_from_encoder_outputs is a wrapper around the base
# StableDiffusion image encoder which samples from the statistical
# distribution produced by the image encoder,
# rather than taking just the mean (like many other SD applications)
def sample_from_encoder_outputs(outputs):
    mean, logvar = tf.split(outputs, 2, axis=-1)
    logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    std = tf.exp(0.5 * logvar)
    sample = tf.random.normal(tf.shape(mean))
    return mean + std * sample

# produces an embedding for a specified timestep for the diffusion model
def get_timestep_embedding(timestep, dim=320, max_period=10000):
    half = dim // 2
    freqs = tf.math.exp(
        -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    return embedding

# produces a tensor of position IDs for the text encoder
# (which is just a series from [1, MAX_PROMPT_LENGTH])
def get_position_ids():
    return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

def plot_images(images, ep=0):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        #ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        iname = 'image_epoch_' + str(ep) + '_' +str(i) + '.png'
        plt.savefig(os.path.join(SAVE_IMAGE_PATH, 'token/') + iname)

def testds(ds):
    i=0
    for item in ds.take(1):
        images, embeddings = item
        plt.imshow(images)
        plt.axis("off")
        i += 1
        iname = 'image' +  str(i) + '.png'
        plt.savefig(os.path.join(SAVE_IMAGE_PATH, 'test/') + iname)



