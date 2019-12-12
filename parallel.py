import torch
import torch.nn as nn

class BetterDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)