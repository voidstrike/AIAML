import os

from torchvision import datasets
from torch.utils.data import DataLoader

DS_NAME = ['mnist', 'svhn', 'cifar10', 'img_folder']
RAW_MAPPING = [956, 344, 340, 950]


def get_data_loader(ds_name, root, batch_size=64, tfs=None, train=True):
    global DS_NAME

    if ds_name not in DS_NAME:
        raise Exception('Unsupported Data Set.')

    ds = None

    if ds_name == 'mnist':
        ds = datasets.MNIST(os.path.join(root, ds_name), train=train, transform=tfs, download=True)
    elif ds_name == 'svhn':
        ds = datasets.SVHN(os.path.join(root, ds_name), split='train' if train else 'test', download=True,
                           transform=tfs)
    elif ds_name == 'cifar10':
        ds = datasets.CIFAR10(os.path.join(root, ds_name), train=train, transform=tfs, download=True)
    elif ds_name == 'img_folder':
        tmp_path = os.path.join(os.path.join(root, 'imgnet'), 'train' if train else 'test')

        def tgt_tfs(raw_idx):
            global RAW_MAPPING
            return RAW_MAPPING[raw_idx]

        ds = datasets.ImageFolder(tmp_path, transform=tfs, target_transform=tgt_tfs)
    else:
        pass

    return DataLoader(ds, batch_size=batch_size, shuffle=True)
