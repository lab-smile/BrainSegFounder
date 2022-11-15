from torch import nn
from typing import Tuple


class GBPretrainedModel(nn.Module):
    def __init__(self, pretrained_model: nn.Module, in_channels: int, out_channels: int, model_params: Tuple):
        pretrained_in, pretrained_out = model_params
        super().__init__()
        self.in_layer = nn.Conv3d(in_channels, pretrained_in,
                                  1)  # kernel size 1 is the 3d equivalent of 2D fully connected layer
        self.final_layer = nn.Conv3d(pretrained_out, out_channels, 1)
        self.pretrained_model = pretrained_model

    def forward(self, input_):
        pretrained_in = self.in_layer(input_)
        pretrained_out = self.pretrained_model(pretrained_in)
        out = self.final_layer(pretrained_out)
        return out

