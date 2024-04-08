
import tensorflow as tf
import os
import codecs
from utils import *
from keras import backend as ops
import tensorflow_datasets as tfds

class ReadWriteTFRecord:
    def __init__(self, datapath):
        self.datapath = datapath

    def _bytes_feature(self, value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
    def image_example(self, image_string, label):
      image_shape = tf.image.decode_png(image_string).shape

      feature = {
          'height': self._int64_feature(image_shape[0]),
          'width': self._int64_feature(image_shape[1]),
          'depth': self._int64_feature(image_shape[2]),
          'label': self._bytes_feature(label),
          'image_raw': self._bytes_feature(image_string),
      }

      return tf.train.Example(features=tf.train.Features(feature=feature))

    def writetfrecord(self, df, record_file='images.tfrecords'):
        datapath = self.datapath
        tfrecord_path = os.path.join(datapath, record_file)
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
          for index, row in df.iterrows():
            image_string = row['image'].get('bytes')
            label = codecs.encode(row['name'], 'utf-8')
            tf_example = self.image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
            # if index > 10:
            #     break

    def _parse_image_function(self, example_proto):
        # Create a dictionary describing the features.
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.string),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)


    def readtfrecord(self, record_file='images.tfrecords'):
        datapath= self.datapath
        tfrecord_path = os.path.join(datapath, record_file)
        raw_image_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_image_dataset = raw_image_dataset.map(self._parse_image_function)
        return parsed_image_dataset

class PrepareData:
    def __init__(self, dataset):
        self.dataset = dataset

    def preprocess_image(self, data):
        image_raw = data['image_raw']
        image = tf.image.decode_png(image_raw)

        # center crop image - not necessary for this dataset
        # height = tf.shape(image)[0]
        # width  = tf.shape(image)[1]
        #
        # crop_size = tf.minimum(height, width)
        # image = tf.image.crop_to_bounding_box(
        #     image,
        #     (height - crop_size) // 2,
        #     (width - crop_size) // 2,
        #     crop_size,
        #     crop_size,
        # )

        # resize and clip
        # for image downsampling it is important to turn on antialiasing
        image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)

    def create_batch(self, ds):
        dataset = (ds
                   .map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .repeat(dataset_repetitions)
                   .shuffle(10 * batch_size)
                   .batch(batch_size, drop_remainder=True)
                   .prefetch(buffer_size=tf.data.AUTOTUNE))

        return dataset

    def prepare_dataset(self):
        # the validation dataset is shuffled as well, because data order matters
        # for the KID estimation

        dl = 0
        for item in self.dataset.as_numpy_iterator():
            dl += 1

        train_size = int(train_per * dl / 100)
        val_size = int(val_per * dl / 100)

        train_dataset = self.create_batch(self.dataset.take(train_size))
        val_dataset   = self.create_batch(self.dataset.skip(train_size))

        full_dataset = self.dataset.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        return train_dataset, val_dataset, full_dataset

class PrepareFlowerData:
    def __init__(self, split):
        self.split = split

    def preprocess_image(self, data):
        # center crop image
        height = ops.shape(data["image"])[0]
        width = ops.shape(data["image"])[1]
        crop_size = ops.minimum(height, width)
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )

        # resize and clip
        # for image downsampling it is important to turn on antialiasing
        image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
        return ops.clip(image / 255.0, 0.0, 1.0)

    def prepare_dataset(self):
        # the validation dataset is shuffled as well, because data order matters
        # for the KID estimation
        split = self.split
        return (
            tfds.load(dataset_name, split=split, shuffle_files=True)
            .map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .repeat(dataset_repetitions)
            .shuffle(10 * batch_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )


