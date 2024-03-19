import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


class SelfAttentionInteraction(nn.Module):
    def __init__(self, sensor_channel, n_channels):
        super(SelfAttentionInteraction, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        f, g, h = self.query(x), self.key(x), self.value(x)

        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)

        o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
        o = o.permute(0, 2, 1).contiguous()
        return o


class FC(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(channel_in, channel_out)

    def forward(self, x):
        x = self.fc(x)
        return (x)


class TemporalLstm(nn.Module):
    def __init__(self, sensor_channel, filter_num):
        super(TemporalLstm, self).__init__()
        self.lstm = nn.LSTM(filter_num,
                            filter_num,
                            batch_first=True)

    def forward(self, x):
        outputs, h = self.lstm(x)
        return outputs


class KeyToAccModel(nn.Module):
    def __init__(
            self,
            input_shape,
            number_class,

            filter_num,
            nb_conv_layers=4,
            filter_size=5,

            dropout=0.1,

            activation="ReLU",

    ):
        super(KeyToAccModel, self).__init__()

        filter_num_list = [1]
        for i in range(nb_conv_layers - 1):
            filter_num_list.append(filter_num)
        filter_num_list.append(filter_num)

        layers_conv = []
        for i in range(nb_conv_layers):
            in_channel = filter_num_list[i]
            out_channel = filter_num_list[i + 1]
            if i % 2 == 1:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (filter_size, 1), (2, 1)),
                    nn.ReLU(inplace=True),  # ))#,
                    nn.BatchNorm2d(out_channel)))
            else:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (filter_size, 1), (1, 1)),
                    nn.ReLU(inplace=True),  # ))#,
                    nn.BatchNorm2d(out_channel)))
        self.layers_conv = nn.ModuleList(layers_conv)
        downsampling_length = self.get_the_shape(input_shape)

        self.channel_interaction = SelfAttentionInteraction(1, filter_num)

        self.channel_fusion = FC(input_shape[3] * filter_num, '''Todo: Add the right number here''')

        self.temporal_interaction = TemporalLstm(input_shape[3], 2 * filter_num)

        self.dropout = nn.Dropout(dropout)

        # Final Prediciotn of Acc reading
        self.prediction = nn.Linear(2 * filter_num, number_class)

    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)

        for layer in self.layers_conv:
            x = layer(x)

        return x.shape[2]

    def forward(self, x):
        # B F L C
        for layer in self.layers_conv:
            x = layer(x)

        x = x.permute(0, 3, 2, 1)
        # ------->  B x C x L* x F*

        """ =============== cross channel interaction ==============="""
        x = torch.cat(
            [self.channel_interaction(x[:, :, t, :]).unsqueeze(3) for t in range(x.shape[2])],
            dim=-1,
        )
        # ------->  B x C x F* x L*

        x = self.dropout(x)

        """=============== cross channel fusion ==============="""

        if self.cross_channel_aggregation_type == "FC":

            x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = self.activation(self.channel_fusion(x))
        elif self.cross_channel_aggregation_type in ["SFCC", "SFCF", "SFCF2"]:
            x = x.permute(0, 3, 1, 2)
            x = self.activation(self.channel_fusion(x))
        else:
            x = torch.cat(
                [self.channel_fusion(x[:, :, :, t]).unsqueeze(2) for t in range(x.shape[3])],
                dim=-1,
            )
            x = x.permute(0, 2, 1)
            x = self.activation(x)
        # ------->  B x L* x F*

        """cross temporal interaction """
        x = self.temporal_interaction(x)

        """cross temporal fusion """
        if self.temporal_info_aggregation_type == "FC":
            x = self.flatten(x)
            x = self.activation(self.temporal_fusion(x))  # B L C
        else:
            x = self.temporal_fusion(x)

        y = self.prediction(x)
        return y
