import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
from PIL import Image
import random
import shutil
from glob import glob
import oci
import zipfile

def donwload_data_oci(namespace, bucket, file, resource_principal=True):
    
    if resource_principal:
    
        # Initialize a Resource Principals provider
        auth_provider = oci.auth.signers.get_resource_principals_signer()
        # Create an Object Storage client
        object_storage_client = oci.object_storage.ObjectStorageClient({}, signer=auth_provider)

    # Get the object
    response = object_storage_client.get_object(namespace, bucket, file)
    
    path_compressed = 'compressed_data'
    path_uncompressed = 'data'
    
    # Save the object data to a file
    with open('compressed_data', 'wb') as file:
        for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
            file.write(chunk)
            
    try:
        with zipfile.ZipFile(path_compressed, 'r') as zip_ref:
            zip_ref.extractall(path_uncompressed)
            print(f"Extracted all files to {path_uncompressed}")
    except FileNotFoundError:
        print(f"Error: The file {path_compressed} was not found.")
    except zipfile.BadZipFile:
        print(f"Error: The file {path_compressed} is not a zip file or it is corrupted.")

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




def split_dataset_digi_face(base_folder, validation_percentage, num_identification_imgs):
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



def create_shards(path_to_data, batch_size, img_dim, num_shards, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    data_transforms = transforms.Compose([transforms.Resize([img_dim, img_dim]),
                                            transforms.ToTensor()])

    # Load the dataset
    data = ImageFolder(root=path_to_data, transform=data_transforms)
    
    # Create a data loader
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    # Calculate total number of samples and the size of each shard
    num_total_samples = len(data)
    shard_size = num_total_samples // num_shards
    remaining = num_total_samples % shard_size
        
    # Create tensors to hold the current shard data
    tensor_data = torch.zeros((shard_size, 3, img_dim, img_dim))
    tensor_labels = torch.zeros(shard_size, dtype=torch.long)
    
    current_shard_idx = 0
    current_shard_count = 0

    batch_iterator = tqdm(data_loader, desc=f"Creating shards")

    for img, lab in batch_iterator:

        batch_size_current = img.size(0)
        
        if current_shard_count + batch_size_current <= shard_size:
            # If the current batch fits into the current shard
            tensor_data[current_shard_count:current_shard_count + batch_size_current] = img
            tensor_labels[current_shard_count:current_shard_count + batch_size_current] = lab
            current_shard_count += batch_size_current
        else:

            # Split the current batch across the current and the next shard
            remaining_space = shard_size - current_shard_count
            # Fill the current shard
            if remaining_space > 0:
                tensor_data[current_shard_count:current_shard_count + remaining_space] = img[:remaining_space]
                tensor_labels[current_shard_count:current_shard_count + remaining_space] = lab[:remaining_space]
            
            save_shard(tensor_data.clone(), tensor_labels.clone(), output_folder, current_shard_idx)
            
            # Start a new shard with the remaining data
            if current_shard_idx == num_shards - 1:
                tensor_data = torch.zeros((shard_size + remaining, 3, img_dim, img_dim))
                tensor_labels = torch.zeros(shard_size + remaining, dtype=torch.long)

            tensor_data = torch.zeros((shard_size, 3, img_dim, img_dim))
            tensor_labels = torch.zeros(shard_size, dtype=torch.long)
            
            current_shard_idx += 1
            current_shard_count = batch_size_current - remaining_space
            
            tensor_data[:current_shard_count] = img[remaining_space:]
            tensor_labels[:current_shard_count] = lab[remaining_space:]


        # Save the last shard if it has remaining data
    if current_shard_count > 0:
        save_shard(tensor_data[:current_shard_count].clone(), tensor_labels[:current_shard_count].clone(), output_folder, current_shard_idx)

    
    return True

def save_shard(images, labels, output_folder, shard_idx):    
    # Save both images and labels in a single file
    dataset = TensorDataset(images, labels)
    shard_file = os.path.join(output_folder, f'shard_{shard_idx}.pt')
    torch.save(dataset, shard_file)


def load_shard(filename):
    dataset = torch.load(filename)
    images, labels = dataset.tensors
    return images, labels


class CustomDataLoader:
    def __init__(self, B, data_root, process_rank, num_processes):
        self.B = B
        self.process_rank = process_rank
        self.num_processes = num_processes

        shards = os.listdir(data_root)
        shards = [os.path.join(data_root, s) for s in shards if s.endswith('.pt')]
        self.shards = shards

        self.current_shard = 0
        self.imgs, self.labels = load_shard(self.shards[self.current_shard])
        self.current_position = self.B * self.process_rank


    def next_batch(self):
        B = self.B

        x = self.imgs[self.current_position : self.current_position + B]
        y = self.labels[self.current_position : self.current_position + B]

        self.current_position += B * self.num_processes

        if self.current_position + (B * self.num_processes) > len(self.labels):

            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.imgs, self.labels = load_shard(self.shards[self.current_shard])
            self.current_position = self.B * self.process_rank

        return x, y





def count_files_in_directory(directory_path):
    """
    Counts the number of files in a directory and its subdirectories.
    
    Parameters:
    directory_path (str): Path to the directory.
    
    Returns:
    int: Number of files in the directory and its subdirectories.
    """
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        file_count += len(files)
    return file_count

# Example usage
directory_path = 'data_unzip'
number_of_files = count_files_in_directory(directory_path)
print(f"Number of files in '{directory_path}' and its subdirectories: {number_of_files}")

