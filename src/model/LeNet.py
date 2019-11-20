import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # Standard LeNet Implementation
    def __init__(self, single=False, attn_output=False):
        super(LeNet, self).__init__()
        if single:
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        else:
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.attn_flag = attn_output

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        attn = torch.mean(abs(out), dim=1).unsqueeze(1)
        # attn = torch.mean(torch.clamp(out, min=0.0), dim=1).unsqueeze(1)

        out = out.flatten(start_dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        if self.attn_flag:
            return out, attn
        else:
            return out
