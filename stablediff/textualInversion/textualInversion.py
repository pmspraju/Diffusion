#Textual inversion works by learning a token embedding for a new text token,
# keeping the remaining components of StableDiffusion frozen.

import sys
import tensorflow as tf
import tensorflow.keras as keras
import keras_cv
import numpy as np
from utils import *
from prepareData import CreateDataset
from stableDiff import StableDiffusionFineTuner
from keras_cv.src.models.stable_diffusion.stable_diffusion import MAX_PROMPT_LENGTH
from keras_cv.models.stable_diffusion import NoiseScheduler
from callbacks import GenerateImages
from tensorflow.keras import callbacks

stable_diffusion = keras_cv.models.StableDiffusion()
# generated = stable_diffusion.text_to_image(
#     f"an oil painting of cat", seed=1337, batch_size=3
# )
#
# # what StableDiffusion produces for our token
# plot_images(generated)
#
# sys.exit(0)
stable_diffusion.tokenizer.add_tokens(PLACEHOLDER_TOKEN)

#dataset = CreateDataset(URLS_SINGLE, PROMPTS_SINGLE, stable_diffusion)
dataset = CreateDataset(LOCAL_SINGLE, PROMPTS_SINGLE, stable_diffusion)
#single_ds = dataset.assemble_image_dataset()
single_ds = dataset.assemble_dataset()

#  Uncomment to test the dataset
#testds(single_ds)
#sys.exit(0)

#dataset = CreateDataset(URLS_GROUP, PROMPTS_GROUP, stable_diffusion)
dataset = CreateDataset(LOCAL_GROUP, PROMPTS_GROUP, stable_diffusion)
group_ds = dataset.assemble_dataset()

#  Uncomment to test the dataset
#testds(group_ds)
#sys.exit(0)

train_ds = single_ds.concatenate(group_ds)
train_ds = train_ds.batch(1).shuffle(
    train_ds.cardinality(), reshuffle_each_iteration=True
)

# concatenate the two datasets
train_ds = single_ds.concatenate(group_ds)
train_ds = train_ds.batch(1).shuffle(
    train_ds.cardinality(), reshuffle_each_iteration=True
)

## Adding a new token to the text encoder

# The embedding layer is the 2nd layer in the text encoder
old_token_weights = stable_diffusion.text_encoder.layers[2].token_embedding.get_weights()
old_token_weights = old_token_weights[0]

old_position_weights = stable_diffusion.text_encoder.layers[2].position_embedding.get_weights()

# Get len of .vocab instead of tokenizer
new_vocab_size = len(stable_diffusion.tokenizer.vocab)

tokenized_initializer = stable_diffusion.tokenizer.encode("cat")[1]
new_weights = stable_diffusion.text_encoder.layers[2].token_embedding(
    tf.constant(tokenized_initializer)
)

new_weights = np.expand_dims(new_weights, axis=0)
new_weights = np.concatenate([old_token_weights, new_weights], axis=0)

## construct a new TextEncoder and prepare it.

# Have to set download_weights False so we can init (otherwise tries to load weights)
new_encoder = keras_cv.models.stable_diffusion.TextEncoder(
    keras_cv.src.models.stable_diffusion.stable_diffusion.MAX_PROMPT_LENGTH,
    vocab_size=new_vocab_size,
    download_weights=False,
)

# Copy the weights from the old encoder to the new one
for index, layer in enumerate(stable_diffusion.text_encoder.layers):
    # Layer 2 is the embedding layer, so we omit it from our weight-copying
    if index == 2:
        continue
    new_encoder.layers[index].set_weights(layer.get_weights())

# Set the new weights for the embedding layer
new_encoder.layers[2].token_embedding.set_weights([new_weights])
new_encoder.layers[2].position_embedding.set_weights(old_position_weights)

# Compile the new encoder
stable_diffusion._text_encoder = new_encoder
stable_diffusion._text_encoder.compile(jit_compile=True)

## In TextualInversion, the only piece of the model that is trained is the embedding vector
## for the new token. This is done by freezing all layers of the text encoder except for the

stable_diffusion.diffusion_model.trainable = False
stable_diffusion.decoder.trainable = False
stable_diffusion.text_encoder.trainable = True

stable_diffusion.text_encoder.layers[2].trainable = True

for layer in traverse_layers(stable_diffusion.text_encoder):
    if isinstance(layer, tf.keras.layers.Embedding) or "clip_embedding" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

new_encoder.layers[2].position_embedding.trainable = False

# Check that only the embedding layer is trainable
# print the weights that were set to trainable
all_models = [
    stable_diffusion.text_encoder,
    stable_diffusion.diffusion_model,
    stable_diffusion.decoder,
]
# print([[w.shape for w in model.trainable_weights] for model in all_models])

# Remove the top layer from the encoder, which cuts off the variance and only returns
# the mean
training_image_encoder = tf.keras.Model(
    stable_diffusion.image_encoder.input,
    stable_diffusion.image_encoder.layers[-2].output,
)

generated = stable_diffusion.text_to_image(
    f"an oil painting of {PLACEHOLDER_TOKEN}", seed=1337, batch_size=3
)

# At this point the model still thinks of our token as a cat,
# as this was the seed token we used to initialize our custom token.

# what StableDiffusion produces for our token
plot_images(generated)

# Fine tune the stable diffusion model for our new token with
# the dataset we created earlier

noise_scheduler = NoiseScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    train_timesteps=1000,
)

learning_rate = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=train_ds.cardinality() * EPOCHS
)
optimizer = keras.optimizers.Adam(
    weight_decay=0.004, learning_rate=learning_rate, epsilon=1e-8, global_clipnorm=10
)

trainer = StableDiffusionFineTuner(stable_diffusion, noise_scheduler, training_image_encoder, name="trainer")

trainer.compile(
    optimizer=optimizer,
    # We are performing reduction manually in our train step, so none is required here.
    loss=keras.losses.MeanSquaredError(reduction="none"),
)


# create three callbacks with different prompts so that
# we can see how they progress over the course of training.
# We use a fixed seed so that we can easily see the progression of the learned token.
checkpoint_path = os.path.join(CHECK_POINT_PATH, "textual_inversion.weights.h5")
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

cbs = [
    GenerateImages(
        stable_diffusion, prompt=f"an oil painting of {PLACEHOLDER_TOKEN}", seed=1337
    ),
    GenerateImages(
        stable_diffusion, prompt=f"gandalf the gray as a {PLACEHOLDER_TOKEN}", seed=1337
    ),
    GenerateImages(
        stable_diffusion,
        prompt=f"two {PLACEHOLDER_TOKEN} getting married, photorealistic, high quality",
        seed=1337,
    ),

    checkpoint_callback,
]

modelPath = os.path.join(CHECK_POINT_PATH, "textual_inversion.weights.h5")

modelexists = False
if os.path.exists(checkpoint_path):
    trainer.load_weights(checkpoint_path, skip_mismatch=True)
    modelexists = True
else:
    # Train the model
    trainer.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=cbs,
    )

if modelexists:
    # generated = trainer.stable_diffusion.text_to_image(
    #     f"Gandalf as a {PLACEHOLDER_TOKEN} fantasy art drawn by disney concept artists, "
    #     "golden colour, high quality, highly detailed, elegant, sharp focus, concept art, "
    #     "character concepts, digital painting, mystery, adventure",
    #     batch_size=3,
    # )
    # generated = trainer.stable_diffusion.text_to_image(
    #     f"A masterpiece of a {PLACEHOLDER_TOKEN} crying out to the heavens. "
    #     f"Behind the {PLACEHOLDER_TOKEN}, an dark, evil shade looms over it - sucking the "
    #     "life right out of it.",
    #     batch_size=3,
    # )
    generated = stable_diffusion.text_to_image(
        f"An evil {PLACEHOLDER_TOKEN}.", batch_size=3
    )
    plot_images(generated)
