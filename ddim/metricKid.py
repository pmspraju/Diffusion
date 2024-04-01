import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *
from keras import backend as ops
import numpy as np

@keras.saving.register_keras_serializable()
class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = ops.cast(ops.shape(features_1)[1], dtype="float32")
        #feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype="float32")
        return (
            features_1 @ ops.transpose(features_2) / feature_dimensions + 1.0
            #features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0
        ) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)
        print('write0')
        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)
        print('write1')
        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = real_features.shape[0]

        batch_size_f = ops.cast(batch_size, dtype="float32")
        #batch_size_f = tf.cast(batch_size, dtype="float32")
        print('write2')
        # mean_kernel_real = ops.sum(kernel_real * (1.0 - ops.eye(batch_size))) / (
        #     batch_size_f * (batch_size_f - 1.0)
        # )
        mean_kernel_real = ops.sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
                batch_size_f * (batch_size_f - 1.0)
        )
        # mean_kernel_real = np.sum(kernel_real * (1.0 - np.eye(batch_size))) / (
        #         batch_size_f * (batch_size_f - 1.0)
        # )
        print('write3')
        # mean_kernel_generated = ops.sum(
        #     kernel_generated * (1.0 - ops.eye(batch_size))
        # ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_generated = ops.sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        print('write4')
        mean_kernel_cross = ops.mean(kernel_cross)
        #mean_kernel_cross = ops.mean(kernel_cross)
        print('write5')
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        print('write6')
        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()
