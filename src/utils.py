import torchvision.transforms as tfs
from art.attacks import DeepFool, CarliniL2Method, CarliniLInfMethod, BasicIterativeMethod
from art.attacks import FastGradientMethod, ProjectedGradientDescent, SaliencyMapMethod
from art.classifiers import PyTorchClassifier

import numpy as np
import torch

D1_SPLIT = ['mnist', 'svhn', 'cifar10']
D2_SPLIT = ['img_folder']

SHAPE_DICT = {
    'mnist': (1, 32),
    'svhn': (3, 32),
    'cifar10': (3, 32),
    'img_folder': (3, 224),
}


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, intensor, c=1):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        intensor[:, 0, :, :] = intensor[:, 0, :, :].mul_(self.std[0]).add_(self.mean[0])
        if c == 3:
            intensor[:, 1, :, :] = intensor[:, 1, :, :].mul_(self.std[1]).add_(self.mean[1])
            intensor[:, 2, :, :] = intensor[:, 2, :, :].mul_(self.std[2]).add_(self.mean[2])

        return intensor


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


def build_adversarial(model, optimizer, loss, input_shape, nb_class, method, batch_size=32, pgd_eps=0.3):
    model.eval()
    wmodel = PyTorchClassifier(model, loss, optimizer, input_shape, nb_class)

    if method == 'deepfool':
        adv_crafter = DeepFool(wmodel)
    elif method == 'bim':
        adv_crafter = BasicIterativeMethod(wmodel, batch_size=batch_size)
    elif method == 'jsma':
        adv_crafter = SaliencyMapMethod(wmodel, batch_size=batch_size)
    elif method == 'cw2':
        adv_crafter = CarliniL2Method(wmodel, batch_size=batch_size)
    elif method == 'cwi':
        adv_crafter = CarliniLInfMethod(wmodel, batch_size=batch_size)
    elif method == 'fgsm':
        adv_crafter = FastGradientMethod(wmodel, batch_size=batch_size)
    elif method == 'pgd':
        adv_crafter = ProjectedGradientDescent(wmodel, batch_size=batch_size, eps=pgd_eps)
    else:
        raise NotImplementedError('Unsupported Attack Method: {}'.format(method))

    return adv_crafter


def generate_adv_tensor(advcrafter, in_tensor):
    adv_np = advcrafter.generate(in_tensor.numpy())
    return torch.from_numpy(adv_np)


def get_shape(name):
    global SHAPE_DICT
    return SHAPE_DICT[name]


