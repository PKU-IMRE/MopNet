import torch
from torch import nn
from NEDB import NEDB
from RNEDB import RNEDB


class NLEDN(nn.Module):
    def __init__(self):
        super(NLEDN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.up_1 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[8, 8])
        self.up_2 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[4, 4])
        self.up_3 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[2, 2])

        self.down_3 = NEDB(block_num=4, inter_channel=32, channel=64)
        self.down_2 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[2, 2])
        self.down_1 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[4, 4])

        self.down_2_fusion = nn.Conv2d(64 + 64, 64, 1, 1, 0)
        self.down_1_fusion = nn.Conv2d(64 + 64, 64, 1, 1, 0)

        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 1, 1, 0),
            nn.Conv2d(64, 64, 3, 1, 1),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        feature_neg_1 = self.conv1(x)
        feature_0 = self.conv2(feature_neg_1)

        up_1_banch = self.up_1(feature_0)
        up_1, indices_1 = nn.MaxPool2d(2, 2, return_indices=True)(up_1_banch)

        up_2 = self.up_2(up_1)
        up_2, indices_2 = nn.MaxPool2d(2, 2, return_indices=True)(up_2)

        up_3 = self.up_3(up_2)
        up_3, indices_3 = nn.MaxPool2d(2, 2, return_indices=True)(up_3)

        down_3 = self.down_3(up_3)

        down_3 = nn.MaxUnpool2d(2, 2)(down_3, indices_3, output_size=up_2.size())

        down_3 = torch.cat([up_2, down_3], dim=1)
        down_3 = self.down_2_fusion(down_3)
        down_2 = self.down_2(down_3)

        down_2 = nn.MaxUnpool2d(2, 2)(down_2, indices_2, output_size=up_1.size())

        down_2 = torch.cat([up_1, down_2], dim=1)
        down_2 = self.down_1_fusion(down_2)
        down_1 = self.down_1(down_2)
        down_1 = nn.MaxUnpool2d(2, 2)(down_1, indices_1, output_size=feature_0.size())

        down_1 = torch.cat([feature_0, down_1], dim=1)

        cat_block_feature = torch.cat([down_1, up_1_banch], 1)
        feature = self.fusion(cat_block_feature)
        feature = feature + feature_neg_1

        outputs = self.final_conv(feature)

        return outputs
