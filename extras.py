import torchvision.transforms as transforms
import os, shutil
import torchvision.datasets as datasets
import torch.nn as nn

def delete_dir_if_exists(directory):
    """
    Remove a directory if it exists

    dir - Directory to remove
    """

    if os.path.exists(directory): # If directory exists
        shutil.rmtree(directory) # Remove it

def create_dir(directory, delete_already_existing = False):
    """
    Create directory. Deletes and recreate directory if already exists

    Parameter:
    string - directory - name of the directory to create if it does not already exist
    delete_already_existing - Delete directory if already existing
    """

    if delete_already_existing: # If delete directory even if it exists
        delete_dir_if_exists(directory) # Delete directory even if it exists
        os.makedirs(directory) # Create a new directory

    else:
        if not os.path.exists(directory): # If directory does not exist
            os.makedirs(directory) # Create new directory

def get_base_transform(img_size = 32,
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]],
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]):

    normalize = transforms.Normalize(mean, std)

    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transform_train, transform_test

def get_train_test_dataset(dataset, train_transform, test_transform):
    trainset = datasets.ImageFolder(os.path.join(dataset, 'train'), transform=train_transform)
    testset = datasets.ImageFolder(os.path.join(dataset, 'test'), transform=test_transform)

    return trainset, testset

def update_resnet18_no_of_classes(resnet18_model, no_of_classes):
    num_ftrs = resnet18_model.linear.in_features
    resnet18_model.linear = nn.Linear(num_ftrs, no_of_classes)

    return resnet18_model
