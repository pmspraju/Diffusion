# data
dataset_name = "oxford_flowers102"
dataset_repetitions = 5
num_epochs = 80  # train for at least 50 epochs for good results
image_size = 64
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

#split
train_per = 80
val_per = 20

#checkpoint path
checkpoint_path_butterfly = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddim/butterfly/'
save_image_path_butterfly = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddim/butterfly/images/'

checkpoint_path_flower = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddim/flower/'
save_image_path_flower = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddim/flower/images/'