import torchvision.transforms as transforms
import os
import torchvision.datasets as datasets

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
