from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
import torch


class AttnVGG16(nn.Module):
    def __init__(self, pool='max'):
        super(AttnVGG16, self).__init__()
        # Standard VGG16 Implementation w/o BN
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.pool1(res)
        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.pool2(res)
        res = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(res))
        res = F.relu(self.conv3_3(res))
        res = self.pool3(res)
        res = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(res))
        res = F.relu(self.conv4_3(res))
        res = self.pool4(res)
        res = F.relu(self.conv5_1(res))
        res = F.relu(self.conv5_2(res))
        res = F.relu(self.conv5_3(res))
        res = self.pool5(res)

        attn = torch.mean(abs(res), dim=1).unsqueeze(1)
        # attn = torch.mean(torch.clamp(res, min=0.0), dim=1).unsqueeze(1)

        res = res.flatten(start_dim=1)
        final_res = self.classifier(res)

        return final_res, attn

    # Copy the weight & bias from other trained VGG-16 model (Same structure in torch model zoo)
    def weight_copy_(self, tgt=None):
        if tgt is None:
            tgt = vgg16(pretrained=True)

        assert tgt.features is not None

        self.conv1_1.weight = tgt.features[0].weight
        self.conv1_1.bias = tgt.features[0].bias
        self.conv1_2.weight = tgt.features[2].weight
        self.conv1_2.bias = tgt.features[2].bias

        self.conv2_1.weight = tgt.features[5].weight
        self.conv2_1.bias = tgt.features[5].bias
        self.conv2_2.weight = tgt.features[7].weight
        self.conv2_2.bias = tgt.features[7].bias

        self.conv3_1.weight = tgt.features[10].weight
        self.conv3_1.bias = tgt.features[10].bias
        self.conv3_2.weight = tgt.features[12].weight
        self.conv3_2.bias = tgt.features[12].bias
        self.conv3_3.weight = tgt.features[14].weight
        self.conv3_3.bias = tgt.features[14].bias

        self.conv4_1.weight = tgt.features[17].weight
        self.conv4_1.bias = tgt.features[17].bias
        self.conv4_2.weight = tgt.features[19].weight
        self.conv4_2.bias = tgt.features[19].bias
        self.conv4_3.weight = tgt.features[21].weight
        self.conv4_3.bias = tgt.features[21].bias

        self.conv5_1.weight = tgt.features[24].weight
        self.conv5_1.bias = tgt.features[24].bias
        self.conv5_2.weight = tgt.features[26].weight
        self.conv5_2.bias = tgt.features[26].bias
        self.conv5_3.weight = tgt.features[28].weight
        self.conv5_3.bias = tgt.features[28].bias

        self.train()


def main():
    tm = AttnVGG16(pool='max')
    tm.weight_copy_()


if __name__ == "__main__":
    main()
