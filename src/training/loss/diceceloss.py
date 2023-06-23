import torch.nn as nn
import torch
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        print("WARNING: Loss only works for two classes right now.")
        super().__init__()
    
    def forward(self, output:torch.Tensor, target:torch.Tensor):
        assert isinstance(target, torch.Tensor)
        assert output.shape == target.shape, f"Got {output.shape} expected " + str(target.shape)

        smooth = 1e-5  # to avoid division by ze        ro
        intersection = torch.sum(output * target)
        total = torch.sum(output) + torch.sum(target)
        dice = (2. * intersection + smooth) / (total + smooth)
        loss = 1 - dice
        return loss/output.shape[0]
    
    @staticmethod
    def dice_score(output:torch.Tensor, target:torch.Tensor):
        smooth = 1e-5  # to avoid division by zero
        intersection = torch.sum(output * target)
        total = torch.sum(output) + torch.sum(target)
        dice = (2. * intersection + smooth) / (total + smooth)
        return dice