import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.Session(config=config)

print("================================================================")
print("----TensorFlow version:", tf.__version__)
print("----Numpy version:", np.__version__)
print("----Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())
print(tf.config.list_logical_devices('GPU'))
print(tf.config.list_physical_devices('GPU'))
print("================================================================")

class DDPMScheduler:

    def __init__(
            self,
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_diffusion_timesteps = num_diffusion_timesteps

    def _warmup_beta(self, warmup_frac):
        betas = self.beta_end * np.ones(self.num_diffusion_timesteps, dtype=np.float64)
        warmup_time = int(self.num_diffusion_timesteps * warmup_frac)
        betas[:warmup_time] = np.linspace(self.beta_start, self.beta_end, warmup_time, dtype=np.float64)
        return betas

    def get_beta_schedule(self, beta_schedule):
        if beta_schedule == 'quad':
            betas = np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_diffusion_timesteps, dtype=np.float64) ** 2
        elif beta_schedule == 'linear':
            betas = np.linspace(self.beta_start, self.beta_end, self.num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'warmup10':
            betas = self._warmup_beta(0.1)
        elif beta_schedule == 'warmup50':
            betas = self._warmup_beta(0.5)
        elif beta_schedule == 'const':
            betas = self.beta_end * np.ones(self.num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1. / np.linspace(self.num_diffusion_timesteps, 1, self.num_diffusion_timesteps, dtype=np.float64)
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (self.num_diffusion_timesteps,)
        return betas

    def noise_like(self, shape, noise_fn=tf.random.normal, repeat=False, dtype=tf.float32):
        repeat_noise = lambda: tf.repeat(noise_fn(shape=(1, *shape[1:]), dtype=dtype), repeats=shape[0], axis=0)
        noise = lambda: noise_fn(shape=shape, dtype=dtype)
        return repeat_noise() if repeat else noise()

path = r'/home/nachiketa/Documents/Workspaces/data/Stablediffusion/Train'
path = os.path.join(path, 'dog.png')

image_string = tf.io.read_file(path)
image = tf.image.decode_image(image_string)

plt.imshow(image.numpy())
plt.savefig('image.png')

normalized_image = tf.image.per_image_standardization(image)
plt.imshow(normalized_image.numpy())
plt.savefig('normalized_image.png')

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# optimization
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4
image_size = 64

#noises = tf.random.normal(shape=(image_size, image_size, 3))
noises = tf.random.normal(shape=(500, 333, 3))

# sample uniform random diffusion times
diffusion_times = tf.random.uniform(
    shape=(1, 1, 1), minval=0.0, maxval=1.0
)

start_angle = tf.math.acos(max_signal_rate) #tf.keras.ops.cast(tf.keras.ops.arccos(max_signal_rate), "float32")
end_angle = tf.math.acos(min_signal_rate) #tf.keras.ops.cast(tf.keras.ops.arccos(min_signal_rate), "float32")
diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

# angles -> signal and noise rates
signal_rates = tf.math.cos(diffusion_angles)
noise_rates = tf.math.sin(diffusion_angles)

#new_size = [image_size, image_size]
#normalized_image = tf.image.resize(normalized_image, new_size)
#plt.imshow(normalized_image.numpy())
#plt.savefig('resized_image.png')

noisy_image = signal_rates * normalized_image + noise_rates * noises
plt.imshow(noisy_image.numpy())
plt.savefig('noised_image.png')







