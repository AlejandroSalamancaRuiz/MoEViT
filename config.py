

# -----------------------------------------------------------------------------
#                               Original Data Split
# -----------------------------------------------------------------------------

# Change to True if train, val and id_imgs folders need to be created
create_train_val_idimgs_folders = False 
original_data_folder = 'casia-web-face-dataset'
validation_percentage = 0.15, 
num_identification_imgs = 500

# -----------------------------------------------------------------------------
#                               Data Configuration
# -----------------------------------------------------------------------------

img_dim = 112

# Point to train, val and id_imgs folder
train_data_path = '/home/alejandro/Documents/Asesoftware/Experiments/data/casia-web-face-dataset/train'
val_data_path = '/home/alejandro/Documents/Asesoftware/Experiments/data/casia-web-face-dataset/validation'
id_imgs_data_path = '/home/alejandro/Documents/Asesoftware/Experiments/data/casia-web-face-dataset/identification_imgs'

shards_folder = 'shards'

num_train_shards = 4
num_val_shards = 1
num_id_imgs_shards = 1

# -----------------------------------------------------------------------------
#                               Model Configuration
# -----------------------------------------------------------------------------

# Changes need to be done in MoEViTConfig class in MoEViT.py

# -----------------------------------------------------------------------------
#                               Training Configuration
# -----------------------------------------------------------------------------

# Changes need to be done in MoEViTConfig class in MoEViT.py

eval_interval = 1000
val_steps = 50

# Optimization hyperparameters following GPT3 paper
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 
weight_decay = 0.1

# Batch Size in tokens following GPT3 paper specs should be 0.5M
total_batch_size = 524288 # 2**19
B = 16 # micro batch size. Increase or decrease depending on GPU memory!!

id_eval_steps = int(num_identification_imgs / B)