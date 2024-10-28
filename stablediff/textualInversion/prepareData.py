import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from keras_cv import layers as cv_layers
from glob import glob
import os
from utils import *

class CreateDataset:

    def __init__(self, urls, prompts, sdiff):
        self.urls = urls
        self.prompts = prompts
        self.sdiff = sdiff

    def assemble_image_dataset(self):
        urls = self.urls

        # Fetch all remote files
        #files = [tf.keras.utils.get_file(origin=url) for url in urls]

        files = glob(os.path.join(urls,'*.jpg')) + glob(os.path.join(urls,'*.png'))

        # Resize images
        resize = keras.layers.Resizing(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, crop_to_aspect_ratio=True)
        images = [keras.utils.load_img(img) for img in files]
        images = [keras.utils.img_to_array(img) for img in images]
        images = np.array([resize(img) for img in images])

        # The StableDiffusion image encoder requires images to be normalized to the
        # [-1, 1] pixel value range
        images = images / 127.5 - 1

        # Create the tf.data.Dataset
        image_dataset = tf.data.Dataset.from_tensor_slices(images)

        # Shuffle and introduce random noise
        image_dataset = image_dataset.shuffle(50, reshuffle_each_iteration=True)
        image_dataset = image_dataset.map(
            cv_layers.RandomCropAndResize(
                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                crop_area_factor=(0.8, 1.0),
                aspect_ratio_factor=(1.0, 1.0),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        image_dataset = image_dataset.map(
            cv_layers.RandomFlip(mode="horizontal"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return image_dataset

    def pad_embedding(self, embedding):
        stable_diffusion = self.sdiff
        return embedding + (
                [stable_diffusion.tokenizer.end_of_text] * (MAX_PROMPT_LENGTH - len(embedding))
        )

    def assemble_text_dataset(self):
        stable_diffusion = self.sdiff
        prompts = self.prompts

        prompts = [prompt.format(PLACEHOLDER_TOKEN) for prompt in prompts]
        embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in prompts]
        embeddings = [np.array(self.pad_embedding(embedding)) for embedding in embeddings]
        text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
        text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
        return text_dataset

    def assemble_dataset(self):
        urls = self.urls
        prompts = self.prompts

        image_dataset = self.assemble_image_dataset()
        text_dataset = self.assemble_text_dataset()
        # the image dataset is quite short, so we repeat it to match the length of the
        # text prompt dataset
        image_dataset = image_dataset.repeat()
        # we use the text prompt dataset to determine the length of the dataset.  Due to
        # the fact that there are relatively few prompts we repeat the dataset 5 times.
        # we have found that this anecdotally improves results.
        text_dataset = text_dataset.repeat(5)
        return tf.data.Dataset.zip((image_dataset, text_dataset))


