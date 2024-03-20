# import the necessary packages
import pandas as pd
import torch.nn as nn
from einops.layers.torch import Rearrange

pd.options.display.float_format = '{:.2f}'.format


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1D CNN Layers, with 5 filters
            nn.Conv2d(1, 5, (3, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 5, (3, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 5, (3, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 5, (3, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            # output is 17 x 5 x 36
            # flatten to 17 x 180
            Rearrange('b c h w -> b h (c w)'),
            # FC
            nn.Linear(180, 100),
            nn.ReLU(),
            # LSTM
            nn.LSTM(100, 100, 1, batch_first=True),
            # output is 17 x 100
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 45),
            nn.ReLU(),
            # make 17 x 1 x 9
            Rearrange('b h (c w) -> b c h w', c=5, w=9),
            # 1D CNN Layers, with 5 filters
            nn.Conv2d(5, 5, (3, 1), stride=(1, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 5, (3, 1), stride=(1, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 5, (3, 1), stride=(1, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 1, (3, 1), stride=(1, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x, h = self.encoder(x)
        x = self.decoder(x)
        return x
