from Solver import Solver

from model.LeNet import LeNet
from model.Vgg16 import AttnVGG16

import argparse


def main(opt):
    # Initialize Model
    net = None
    if opt.model == 'lenet':
        if opt.dataset == 'mnist':
            net = LeNet(True)
        else:
            net = LeNet()
    elif opt.model == 'vgg16':
        net = AttnVGG16()
    else:
        raise NotImplementedError('Unsupported Model Selected')

    # Initialize Solver
    solver = Solver(net, opt)
    if opt.mode == 'train' or opt.mode == 'mix':
        solver.train(opt.epoch)
        print('Saving Model===============================================')
        solver.save(opt.model_path)
        if opt.mode == 'mix':
            print('Current Model Performance in Test Split====================')
            solver.test()
    else:
        solver.load(opt.model_path)
        if opt.mode == 'test':
            solver.test()
        elif opt.mode == 'attack':
            solver.test_with_attack(opt.attack)
        elif opt.mode == 'attn_attack':
            solver.test_with_attack_and_attention(opt.attack)
        elif opt.mode == 'visualize':
            solver.sample_images_pgd()
            # solver.sample_images()
        else:
            raise NotImplementedError('Unsupported Mode Selected')

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic configuration
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset, mnist|svhn|cifar10|img_folder')
    parser.add_argument('--model', type=str, required=True, help='Name of the model, lenet|vgg16')
    parser.add_argument('--model_path', type=str, default='../models/',
                        help='The root directory that stores trained model')
    parser.add_argument('--sample_path', type=str, default='../samples/')
    parser.add_argument('--mode', type=str, default='train', help='Mode to execute, train|test|attack|attn_attack')
    parser.add_argument('--attack', type=str, default='fgsm', help='Name of the attack method, '
                                                                   'fgsm|deepfool|pgd|cw2|cwi|trojan')
    parser.add_argument('--root', type=str, default='../data/', help='The root directory that stores data corpus.')
    parser.add_argument('--crop_size', type=int, default=36, help='Crop_size for the image, 36|268')
    parser.add_argument('--image_size', type=int, default=32, help='Image size for the image, 32|224')
    parser.add_argument('--display', type=int, default=1)

    # Training parameters
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=.999)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)

    # Attn Comparison params
    parser.add_argument('--thresh', type=float, default=.5)

    opt = parser.parse_args()
    main(opt)

