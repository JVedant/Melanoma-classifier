import torch
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F

class ResNext101_64x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(ResNext101_64x4d, self).__init__()
        self.model = pretrainedmodels.resnext101_64x4d(pretrained=pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        out = torch.sigmoid(self.l0(x))
        loss = nn.BCELoss()(out, targets.view(-1, 1).type_as(x))
        return out, loss