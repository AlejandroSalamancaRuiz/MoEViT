import os
from Data_Utils import create_shards, split_dataset_casia_wf
import config

# -----------------------------------------------------------------------------
#                       Create Train-Val-Identification_Imgs Folders
# -----------------------------------------------------------------------------

if config.create_train_val_idimgs_folders:
    split_dataset_casia_wf(base_folder=config.original_data_folder, 
                        validation_percentage=config.validation_percentage, 
                        num_identification_imgs=config.num_identification_imgs)

# -----------------------------------------------------------------------------
#                               Create Data Shards
# -----------------------------------------------------------------------------

train_shards_folder = os.path.join(config.shards_folder, 'train')
val_shards_folder = os.path.join(config.shards_folder, 'validation')
id_img_shards_folder = os.path.join(config.shards_folder, 'identification_imgs')

## batch of 2048 because not running on GPU!
create_shards(config.train_data_path, 2048, config.img_dim, config.num_train_shards, train_shards_folder) 
create_shards(config.val_data_path, 2048, config.img_dim, config.num_val_shards, val_shards_folder)
create_shards(config.id_imgs_data_path, 2048, config.img_dim, config.num_id_imgs_shards, id_img_shards_folder)

