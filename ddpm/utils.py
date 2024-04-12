# data
batch_size = 32
num_epochs = 1  # Just for the sake of demonstration
total_timesteps = 1000
norm_groups = 8  # Number of groups used in GroupNormalization layer
learning_rate = 2e-4

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

dataset_name = "oxford_flowers102"
splits = ["train"]

#split
train_per = 80
val_per = 20

#checkpoint path
checkpoint_path_butterfly = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddpm/butterfly/'
save_image_path_butterfly = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddpm/butterfly/images/'

checkpoint_path_flower = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddpm/flower/'
save_image_path_flower = r'/home/nachiketa/Documents/Workspaces/checkpoints/ddpm/flower/images/'