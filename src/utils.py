import torchvision.transforms as tfs

D1_SPLIT = ['mnist', 'svhn', 'cifar10']
D2_SPLIT = ['img_folder']


def get_transformer(ds_name, train, crop_size, image_size):
    global D1_SPLIT, D2_SPLIT
    component = list()  # No horizontal flip

    if train:
        component.append(tfs.Resize(crop_size))
        component.append(tfs.RandomCrop(image_size))
    else:
        component.append(tfs.Resize(image_size))

    component.append(tfs.ToTensor())

    if ds_name in D1_SPLIT:
        component.append(tfs.Normalize(mean=(.5, ), std=(.5, )))
    elif ds_name in D2_SPLIT:
        component.append(tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    return tfs.Compose(component)
