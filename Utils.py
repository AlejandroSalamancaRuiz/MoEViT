import os
import shutil
from glob import glob
import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from PIL import Image
import math


def split_dataset_casia_wf(base_folder, validation_percentage, num_identification_imgs):
    """
    Split dataset into training and validation sets for a specific folder structure.

    Parameters:
    - base_folder: The path to the base folder ('Casia-web-face').
    - validation_percentage: The percentage of images to be used for validation.
    """
    # Define the path to the nested dataset directory
    dataset_path = os.path.join(base_folder, 'casia-webface')

    # Create the train and validation directories if they don't exist
    train_dir = os.path.join(base_folder, 'train')
    val_dir = os.path.join(base_folder, 'validation')
    identification_dir = os.path.join(base_folder, 'identification_imgs')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(identification_dir, exist_ok=True)

    c = 0

    # Iterate through each class directory in the dataset
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        
        # Skip if it's not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Create corresponding class directories in train and validation folders
        train_class_dir = os.path.join(train_dir, class_dir)
        val_class_dir = os.path.join(val_dir, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get a list of all images in the class directory
        images = glob(os.path.join(class_path, '*'))
        random.shuffle(images)  # Shuffle the images to ensure random split

        # Calculate the number of images for validation
        val_count = int(len(images) * validation_percentage)

        val_count = 1 if val_count == 0 else val_count

        if val_count > 1 and c < num_identification_imgs:
            ident_class_dir = os.path.join(identification_dir, class_dir)
            os.makedirs(ident_class_dir, exist_ok=True)

        # Split images into training and validation sets
        validation_images = images[:val_count]
        training_images = images[val_count:]

        c2 = 0
        # Copy validation images to the validation directory
        for image in validation_images:
            shutil.copy(image, os.path.join(val_class_dir, os.path.basename(image)))

            if val_count > 1 and c < num_identification_imgs and c2 < 2:
                shutil.copy(image, os.path.join(ident_class_dir, os.path.basename(image)))
                c2+=1
                c+=1

        # Copy training images to the training directory
        for image in training_images:
            shutil.copy(image, os.path.join(train_class_dir, os.path.basename(image)))





def slplit_dataset_digi_face(base_folder, validation_percentage, num_identification_imgs):
    """
    Split dataset into training and validation sets for a specific folder structure.

    Parameters:
    - base_folder: The path to the base folder ('Casia-web-face').
    - validation_percentage: The percentage of images to be used for validation.
    """
    # Define the path to the nested dataset directory
    dataset_path = os.path.join(base_folder, 'original_data')

    # Create the train and validation directories if they don't exist
    train_dir = os.path.join(base_folder, 'train')
    val_dir = os.path.join(base_folder, 'validation')
    identification_dir = os.path.join(base_folder, 'identification_imgs')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(identification_dir, exist_ok=True)

    c = 0

    # Iterate through each class directory in the dataset

    for class_dir1 in os.listdir(dataset_path):
        for class_dir in os.listdir(os.path.join(dataset_path, class_dir1)):
            class_path = os.path.join(dataset_path, class_dir1, class_dir)
            
            # Skip if it's not a directory
            if not os.path.isdir(class_path):
                continue
            
            # Create corresponding class directories in train and validation folders
            train_class_dir = os.path.join(train_dir, class_dir)
            val_class_dir = os.path.join(val_dir, class_dir)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            # Get a list of all images in the class directory
            images = glob(os.path.join(class_path, '*'))
            random.shuffle(images)  # Shuffle the images to ensure random split

            # Calculate the number of images for validation
            val_count = int(len(images) * validation_percentage)

            val_count = 1 if val_count == 0 else val_count

            if val_count > 1 and c < num_identification_imgs:
                ident_class_dir = os.path.join(identification_dir, class_dir)
                os.makedirs(ident_class_dir, exist_ok=True)

            # Split images into training and validation sets
            validation_images = images[:val_count]
            training_images = images[val_count:]

            c2 = 0
            # Copy validation images to the validation directory
            for image in validation_images:
                shutil.copy(image, os.path.join(val_class_dir, os.path.basename(image)))

                if val_count > 1 and c < num_identification_imgs and c2 < 2:
                    shutil.copy(image, os.path.join(ident_class_dir, os.path.basename(image)))
                    c2+=1
                    c+=1

            # Copy training images to the training directory
            for image in training_images:
                shutil.copy(image, os.path.join(train_class_dir, os.path.basename(image)))



def create_data_loader(path_to_data, img_dim, batch_size, seed=42):
  
    data_transforms_no_aug = transforms.Compose([
        transforms.Resize([img_dim, img_dim]),
        transforms.ToTensor(),
        transforms.Normalize((0.5201, 0.4043, 0.3465), (0.2809, 0.2440, 0.2354))])
    
    data = ImageFolder(root = path_to_data, transform = data_transforms_no_aug)

    loader = DataLoader(data, shuffle=True, batch_size = batch_size)

    return loader


def create_unseen_embds(model, id_imgs_loader, device):

    model.to(device)
    embd_cpu, labels_cpu = None, None
    start = True

    for images, labels in id_imgs_loader:
        images, labels = images.to(device), labels.to(device)
        logits, embeddings, loss = model(images, labels)

        if start:
            labels_cpu = labels.to('cpu')
            embd_cpu = embeddings.to('cpu')
            start = False
        else:
            embd_cpu = torch.cat((embd_cpu, embeddings.to('cpu')), dim=0)
            labels_cpu = torch.cat((labels_cpu, labels.to('cpu')))


    return embd_cpu, labels_cpu


def rank_acc(embeddings, labels):

    # Compute cosine similarity using efficient matrix operations
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarities = torch.mm(embeddings_norm, embeddings_norm.t())
    similarities.fill_diagonal_(0)  # Remove self-similarity
    
    # Sort and get indices of top matches
    sorted_indices = similarities.argsort(descending=True)

    rank_1_acc = 0.0
    rank_5_acc = 0.0

    for i in range(5):
        rank_5_acc += (labels[sorted_indices[:, i]] == labels).sum().item() 
        if i == 0:
            rank_1_acc = rank_5_acc
        
    return rank_1_acc / labels.shape[0], rank_5_acc / labels.shape[0]


def get_lr(it, warmup_steps, min_lr, max_lr, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)