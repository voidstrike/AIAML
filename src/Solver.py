from .dataloader import get_data_loader
from .utils import *
from torch import nn
from torch.optim import Adam

import torch
import os


class Solver(object):
    def __init__(self, model, conf):
        self.model = model
        self.opt = conf

        if self.opt.mode == 'train' or self.opt.mode == 'mix':
            train_tfs = get_transformer(self.opt.dataset, True, crop_size=self.opt.crop_size,
                                        image_size=self.opt.image_size)
            self.train_dl = get_data_loader(self.opt.dataset, self.opt.root, self.opt.batch_size, train_tfs, True)

        if self.opt.mode == 'test' or self.opt.mode == 'mix':
            test_tfs = get_transformer(self.opt.dataset, False, crop_size=None, image_size=self.opt.image_size)
            self.test_dl = get_data_loader(self.opt.dataset, self.opt.root, self.opt.batch_size, test_tfs, False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

        self.gpu_ids = self.opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

    def train(self, epoch):
        self.model.train()
        print('Start Training Process ======================================')
        for idx in range(1, epoch + 1):
            print('Working on epoch {}' % idx)
            hits, total = 0., 0.
            train_loss = 0.
            for feature, label in self.train_dl:
                feature = feature.to(self.device)
                label = label.to(self.device)

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
                tmp_acc = 100 * hits / total
                tmp_loss = train_loss / total
                print('Training Accuracy : {}% and Training Loss : {}' % tmp_acc, tmp_loss)

    def test(self):
        pass

    def save(self, tgt_path):
        torch.save(self.model.state_dict(), os.path.join(tgt_path, str(self.opt.dataset) + '_G.pt'))

    def load(self, tgt_path):
        param_corpus = torch.load(os.path.join(tgt_path, str(self.opt.dataset) + '_G.pt'))
        self.model.load_state_dict(param_corpus)
        pass






