import torch
import numpy as np
from torch import nn


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        # Modified by Riheng
        # self.i = 0
        self.register_buffer('i', torch.IntTensor([1]))
        self.drop_values = np.linspace(
            start=start_value, stop=stop_value, num=nr_steps)
        # self.drop_values = torch.from_numpy(np.linspace(start=start_value, stop=stop_value, num=nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i[0] < len(self.drop_values):
            # if self.i.data[0] < self.drop_values.shape[0]:
            self.dropblock.drop_prob[0] = self.drop_values[self.i[0]]
            #print('{}: {:.6f}'.format(self.i.data[0], self.dropblock.drop_prob))
        self.i.add_(1)
