from dataloader import get_data_loader
from utils import *
from torch import nn
from torch.optim import Adam
from utils import generate_adv_tensor, build_adversarial, get_shape
from PIL import Image

import torch
import os
import statistics
import cv2

import numpy as np


class Solver(object):
    def __init__(self, model, conf):
        stuple = get_shape(conf.dataset)
        self.input_shape = (conf.batch_size, stuple[0], stuple[1], stuple[1])
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.opt = conf
        self.default_root = os.getcwd()
        self.sample_path =os.path.join(self.default_root, conf.sample_path)

        if self.opt.mode == 'train' or self.opt.mode == 'mix':
            train_tfs = get_transformer(self.opt.dataset, True, crop_size=self.opt.crop_size,
                                        image_size=self.opt.image_size)
            self.train_dl = get_data_loader(self.opt.dataset, os.path.join(self.default_root, self.opt.root),
                                            self.opt.batch_size, train_tfs, True)

        test_tfs = get_transformer(self.opt.dataset, False, crop_size=None, image_size=self.opt.image_size)
        self.test_dl = get_data_loader(self.opt.dataset, os.path.join(self.default_root, self.opt.root), self.opt.batch_size, test_tfs, False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

        self.num_classes = 1000 if self.opt.dataset == 'img_folder' else 10

    def train(self, epoch):
        if self.opt.dataset == 'img_folder':
            self.model.weight_copy_()
            self.model.mutate_clf(True)

        self.model.train()
        print('Start Training Process ======================================')
        self.model.attn_flag = True
        for idx in range(1, epoch + 1):
            print('Working on epoch {}'.format(str(idx)))
            hits, total = 0., 0.
            train_loss = 0.
            for feature, label in self.train_dl:
                if torch.cuda.is_available():
                    label = label.cuda()
                    feature = feature.cuda()

                # feature = Variable(feature)
                # label = Variable(label)

                self.optimizer.zero_grad()
                pred, _ = self.model(feature)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()

                # Code for ACC.
                _, pred_idx = torch.max(pred.data, 1)
                total += label.size(0)
                hits += pred_idx.eq(label.data).cpu().sum().float()

                train_loss += loss.item()

            if idx % self.opt.display == 0:
                tmp_acc = 100 * (hits.item()) / total
                tmp_loss = train_loss / total
                print('Training Accuracy : {}% and Training Loss : {}'.format(tmp_acc, tmp_loss))

    def test(self, dl=None):
        self.model.eval()
        self.model.attn_flag = True
        print('Start Testing Process ======================================')

        test_dl = self.test_dl if dl is None else dl

        hits, total = 0., 0.
        train_loss = 0.
        for feature, label in test_dl:
            if torch.cuda.is_available():
                label = label.cuda()
                feature = feature.cuda()

            pred, _ = self.model(feature)
            loss = self.criterion(pred, label)

            # Code for ACC.
            _, pred_idx = torch.max(pred.data, 1)
            total += label.size(0)
            hits += pred_idx.eq(label.data).cpu().sum().float()

            train_loss += loss.item()

        tmp_acc = 100 * (hits.item()) / total
        tmp_loss = train_loss / total
        print('Testing Accuracy : {}% and Test Loss (CrossEntropy) : {}'.format(tmp_acc, tmp_loss))

    def test_with_attack(self, attack, dl=None):
        # self.model.eval()
        # model.attn_flag =
        print('Start Attacking Process ======================================')

        test_dl = self.test_dl if dl is None else dl
        adv_crafter = build_adversarial(self.model, self.optimizer, self.criterion, self.input_shape, self.num_classes,
                                        attack, self.opt.batch_size)

        score_list_acc = list()
        score_list_cn = list()
        for _ in range(10):
            hits, total = 0., 0.
            train_loss = 0.
            for feature, label in test_dl:
                if feature.size(0) != self.opt.batch_size:
                    continue

                self.model.attn_flag = False
                feature = generate_adv_tensor(adv_crafter, feature)
                self.model.attn_flag = True

                if torch.cuda.is_available():
                    label = label.cuda()
                    feature = feature.cuda()

                pred, _ = self.model(feature)
                loss = self.criterion(pred, label)

                # Code for ACC.
                _, pred_idx = torch.max(pred.data, 1)
                total += label.size(0)
                hits += pred_idx.eq(label.data).cpu().sum().float()

                train_loss += loss.item()

            tmp_acc = 100 * (hits.item()) / total
            tmp_loss = train_loss / total
            print('Attack Acc. Using {} : {}% and CrossEntropy Loss : {}'.format(self.opt.attack, tmp_acc, tmp_loss))
            score_list_acc.append(tmp_acc)
            score_list_cn.append(tmp_loss)
        self.get_mean_std(score_list_acc, 'Accuracy')
        self.get_mean_std(score_list_cn, 'Cross Entropy')

    #  batch_size must be 1 in this mode 'attn_test'
    def test_with_attack_and_attention(self, attack, dl=None):
        self.model.eval()
        # model.attn_flag =
        print('Start Attacking Process ======================================')
        if self.opt.batch_size != 1:
            raise Exception('Batch Size in this mode must be 1')

        test_dl = self.test_dl if dl is None else dl
        adv_crafter = build_adversarial(self.model, self.optimizer, self.criterion, self.input_shape, self.num_classes,
                                        attack, self.opt.batch_size)

        total_aaad, total_remain_ratio = 0., 0.
        total_count = 0.
        for feature, _ in test_dl:
            total_count += 1

            self.model.attn_flag = False
            adv_feature = generate_adv_tensor(adv_crafter, feature)
            self.model.attn_flag = True

            if torch.cuda.is_available():
                feature = feature.cuda()
                adv_feature = adv_feature.cuda()

            # Attn map is a (1, 1, un, un) tensor, actual prediction is dropped
            _, adv_attn_map = self.model(adv_feature)
            _, attn_map = self.model(feature)

            adv_attn_map = self.minmax_norm(adv_attn_map)
            attn_map = self.minmax_norm(attn_map)

            _, _, _, tmp_size = attn_map.shape

            aaad = torch.mean(abs(adv_attn_map - attn_map)) / tmp_size ** 2
            total_aaad += aaad.item()

            attn_map_threshold_mask = (attn_map > self.opt.thresh).float()
            attended_area = attn_map_threshold_mask.sum()
            adv_attn_map_mask = (adv_attn_map * adv_attn_map > self.opt.thresh).float()
            remaining_area = adv_attn_map_mask.sum()

            remain_ratio = remaining_area / attended_area
            total_remain_ratio += remain_ratio.item()

        fin_aaad = total_aaad / total_count
        fin_rratio = 100 * total_remain_ratio / total_count
        print('Attack Attention Remaining Ratio : {}% and Average Attention Difference per-pixel : {}'.format(fin_rratio, fin_aaad))

        #  batch_size must be 1 in this mode 'attn_test'

    def sample_images(self, dl=None):
        self.model.eval()
        # model.attn_flag =
        print('Start Image Sampling ======================================')
        if self.opt.batch_size != 1:
            raise Exception('Batch Size in this mode must be 1')

        test_dl = self.test_dl if dl is None else dl
        attack_list = ['fgsm', 'pgd', 'cw2', 'cwi', 'bim']

        if self.opt.dataset != 'img_folder':
            unnorm = UnNormalize((.5, .5, .5), (.5, .5, .5))
        else:
            unnorm = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # unnorm = UnNormalize((.5, .5, .5), (.5, .5, .5))

        for idx, (feature, _) in enumerate(test_dl):
            if idx > 9:
                continue

            print('Working on visualization for {}-th image'.format(idx+1))
            if self.input_shape[1] == 1:
                src_img = unnorm(feature, c=1)
                src_img = src_img.cpu().repeat(1, 3, 1, 1).squeeze(0)
            else:
                src_img = unnorm(feature, c=3)
                src_img = src_img.cpu().squeeze(0)

            src_img = np.transpose(src_img, (1, 2, 0)) * 255.0
            src_img = cv2.cvtColor(src_img.numpy(), cv2.COLOR_RGB2BGR)
            src_img_v2 = np.copy(src_img)

            #  Compute initial forward pass
            if torch.cuda.is_available():
                feature = feature.cuda()

            self.model.attn_flag = True
            _, attn_map = self.model(feature)
            attn_map = self.minmax_norm(attn_map).squeeze().detach()

            # Attn_map visualization
            attn_heat = attn_map.cpu().numpy()
            attn_heat = cv2.resize(attn_heat, (self.input_shape[2], self.input_shape[2]))
            attn_heat = (255 * attn_heat).astype(np.uint8)
            attn_heat = cv2.applyColorMap(attn_heat, cv2.COLORMAP_JET)
            img_list = np.concatenate((src_img, src_img_v2 + 0.4 * attn_heat), axis=1)

            feature = feature.cpu()

            for eachAttack in attack_list:
                # GO through each attack method, time consuming
                print('Generating {} attention image'.format(eachAttack))
                adv_crafter = build_adversarial(self.model, self.optimizer, self.criterion, self.input_shape,
                                                self.num_classes,
                                                eachAttack, self.opt.batch_size)

                self.model.attn_flag = False
                adv_feature = generate_adv_tensor(adv_crafter, feature)
                self.model.attn_flag = True

                if torch.cuda.is_available():
                    adv_feature = adv_feature.cuda()

                # Attn map is a (1, 1, un, un) tensor, actual prediction is dropped
                _, adv_attn_map = self.model(adv_feature)
                adv_attn_map = self.minmax_norm(adv_attn_map).squeeze().detach()
                # adv_attn_map = nn.functional.interpolate(adv_attn_map, (self.input_shape[2], self.input_shape[2]),
                #                                          mode='bilinear')

                # Adv Attn_map visualization
                adv_attn_heat = adv_attn_map.cpu().numpy()
                adv_attn_heat = cv2.resize(adv_attn_heat, (self.input_shape[2], self.input_shape[2]))
                adv_attn_heat = (255 * adv_attn_heat).astype(np.uint8)
                adv_attn_heat = cv2.applyColorMap(adv_attn_heat, cv2.COLORMAP_JET)
                img_list = np.concatenate((img_list, src_img_v2 + 0.4 * adv_attn_heat), axis=1)

            cv2.imwrite(os.path.join(self.sample_path, '{}_{}.png'.format(self.opt.dataset, idx+1)), img_list)

    def sample_images_pgd(self, dl=None):
        self.model.eval()
        # model.attn_flag =
        print('Start Image Sampling ======================================')
        if self.opt.batch_size != 1:
            raise Exception('Batch Size in this mode must be 1')

        test_dl = self.test_dl if dl is None else dl

        if self.opt.dataset != 'img_folder':
            unnorm = UnNormalize((.5, .5, .5), (.5, .5, .5))
        else:
            unnorm = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        for idx, (feature, _) in enumerate(test_dl):
            if idx > 9:
                continue

            print('Working on visualization for {}-th image'.format(idx+1))
            if self.input_shape[1] == 1:
                src_img = unnorm(feature, c=1)
                src_img = src_img.cpu().repeat(1, 3, 1, 1).squeeze(0)
            else:
                src_img = unnorm(feature, c=3)
                src_img = src_img.cpu().squeeze(0)

            src_img = np.transpose(src_img, (1, 2, 0)) * 255.0
            src_img = cv2.cvtColor(src_img.numpy(), cv2.COLOR_RGB2BGR)
            src_img_v2 = np.copy(src_img)

            #  Compute initial forward pass
            if torch.cuda.is_available():
                feature = feature.cuda()

            self.model.attn_flag = True
            _, attn_map = self.model(feature)
            attn_map = self.minmax_norm(attn_map).squeeze().detach()

            # Attn_map visualization
            attn_heat = attn_map.cpu().numpy()
            attn_heat = cv2.resize(attn_heat, (self.input_shape[2], self.input_shape[2]))
            attn_heat = (255 * attn_heat).astype(np.uint8)
            attn_heat = cv2.applyColorMap(attn_heat, cv2.COLORMAP_JET)
            img_list = np.concatenate((src_img, src_img_v2 + 0.4 * attn_heat), axis=1)

            feature = feature.cpu()

            for pgd_eps in range(1, 10, 2):
                # GO through each attack method, time consuming
                tmp_eps = pgd_eps / 10
                print('Generating PGD-{} attention image'.format(tmp_eps))
                adv_crafter = build_adversarial(self.model, self.optimizer, self.criterion, self.input_shape,
                                                self.num_classes,
                                                'pgd', self.opt.batch_size, tmp_eps)

                self.model.attn_flag = False
                adv_feature = generate_adv_tensor(adv_crafter, feature)
                self.model.attn_flag = True

                if torch.cuda.is_available():
                    adv_feature = adv_feature.cuda()

                # Attn map is a (1, 1, un, un) tensor, actual prediction is dropped
                _, adv_attn_map = self.model(adv_feature)
                adv_attn_map = self.minmax_norm(adv_attn_map).squeeze().detach()
                # adv_attn_map = nn.functional.interpolate(adv_attn_map, (self.input_shape[2], self.input_shape[2]),
                #                                          mode='bilinear')

                # Adv Attn_map visualization
                adv_attn_heat = adv_attn_map.cpu().numpy()
                adv_attn_heat = cv2.resize(adv_attn_heat, (self.input_shape[2], self.input_shape[2]))
                adv_attn_heat = (255 * adv_attn_heat).astype(np.uint8)
                adv_attn_heat = cv2.applyColorMap(adv_attn_heat, cv2.COLORMAP_JET)
                img_list = np.concatenate((img_list, src_img_v2 + 0.4 * adv_attn_heat), axis=1)

            cv2.imwrite(os.path.join(self.sample_path, '{}_{}_pgd.png'.format(self.opt.dataset, idx+1)), img_list)

    def save(self, tgt_path):
        tgt_path = os.path.join(self.default_root, tgt_path)
        tgt_path += str(self.opt.dataset) + '_G.pt'
        torch.save(self.model.state_dict(), tgt_path)

    def load(self, tgt_path):
        param_corpus = torch.load(os.path.join(tgt_path, str(self.opt.dataset) + '_G.pt'))
        self.model.load_state_dict(param_corpus)
        pass

    @staticmethod
    def get_mean_std(score_list, name):
        print('Statisitc Information for {}:'.format(name))
        print('Mean : {}, Std: {}'.format(statistics.mean(score_list), statistics.stdev(score_list)))

    @staticmethod
    def minmax_norm(in_tensor):
        t_max, t_min = torch.max(in_tensor), torch.min(in_tensor)
        return (in_tensor - t_min) / (t_max - t_min)







