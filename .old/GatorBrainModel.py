import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act)
    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), act)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act)

    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class GatorBrainModel(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, base_model, n_class=1, act='relu'):
        super(GatorBrainModel, self).__init__()

        self.base_model = base_model

        # This code modified from MG: MG uses 512 as its lowest transition
        # while swinVIT requires the input to be divisible by 12 (512 is not)
        self.up_tr384 = UpTransition(384, 384, 3, act)
        self.up_tr96 = UpTransition(96, 96, 2, act)
        self.up_tr48 = UpTransition(48, 48, 1, act)
        self.up_tr24 = UpTransition(48, 48, 0, act)
        self.out_tr = OutputTransition(24, n_class)

    def forward(self, x_in):
        hidden_states_out = self.base_model.swinVIT(x_in, self.base_model.normalize)
        enc0 = self.base_model.encoder1(x_in)  # 24
        enc1 = self.base_model.encoder2(hidden_states_out[0])  # 24
        enc2 = self.base_model.encoder3(hidden_states_out[1])  # 48
        enc3 = self.base_model.encoder4(hidden_states_out[2])  # 96
        enc4 = self.base_model.encoder10(hidden_states_out[4])  # 384
        # here we switch to modified MG code (not sure the transition works)
        dec4 = self.up_tr384(enc4, hidden_states_out[3])
        dec3 = self.up_tr96(dec4, enc3)
        dec2 = self.up_tr48(dec3, enc2)
        dec1 = self.up_tr24(dec2, enc1)
        dec0 = self.up_tr24(dec1, enc0)
        out = self.out_tr(dec0)
        return out