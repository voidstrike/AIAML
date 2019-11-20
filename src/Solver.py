from dataloader import get_data_loader
from utils import *
from torch import nn
from torch.optim import Adam
from utils import generate_adv_tensor, build_adversarial, get_shape

import torch
import os
import statistics


class Solver(object):
    def __init__(self, model, conf):
        stuple = get_shape(conf.dataset)
        self.input_shape = (conf.batch_size, stuple[0], stuple[1], stuple[1])
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.opt = conf
        self.default_root = os.getcwd()

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

            aaad = torch.mean(abs(adv_attn_map - attn_map)) / self.input_shape[2] ** 2
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







