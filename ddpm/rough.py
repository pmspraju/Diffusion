import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

beta_start=1e-4
beta_end=0.02
timesteps=1000

betas = np.linspace(
        beta_start,
        beta_end,
        timesteps,
        dtype=np.float64,
)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

sqrt_alphas_cumprod = tf.constant(
        np.sqrt(alphas_cumprod), dtype=tf.float32
)

sqrt_one_minus_alphas_cumprod = tf.constant(
        np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
)

log_one_minus_alphas_cumprod = tf.constant(
        np.log(1.0 - alphas_cumprod), dtype=tf.float32
)

sqrt_recip_alphas_cumprod = tf.constant(
        np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
)

sqrt_recipm1_alphas_cumprod = tf.constant(
        np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
)

# Calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
)
posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

posterior_log_variance_clipped = tf.constant(
        np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
)

posterior_mean_coef1 = tf.constant(
        betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        dtype=tf.float32,
)

posterior_mean_coef2 = tf.constant(
    (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
    dtype=tf.float32,
)

graphs = [betas, alphas, alphas_cumprod, alphas_cumprod_prev,
          sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
          log_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod,
          sqrt_recipm1_alphas_cumprod, posterior_variance, posterior_log_variance_clipped,
          posterior_mean_coef1, posterior_mean_coef2]

fig, axs = plt.subplots(2, 7)

k=0
for i in range(2):
    for j in range(7):
        if k < 13:
            axs[i, j].plot(graphs[k])
        k+=1

plt.show()
plt.savefig('variances.png')
