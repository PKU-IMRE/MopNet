import torch
from torch import nn
from non_local_block import NONLocalBlock2D


class NEDB(nn.Module):
    def __init__(self, block_num=3, inter_channel=32, channel=64):
        super(NEDB, self).__init__()

        concat_channels = channel + block_num * inter_channel
        channels_now = channel

        self.non_local = NONLocalBlock2D(channels_now, bn_layer=False)

        self.group_list = []
        for i in range(block_num):
            group = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channel, kernel_size=3,
                          stride=1, padding=1),
                nn.ReLU(),
            )
            self.add_module(name='group_%d' % i, module=group)
            self.group_list.append(group)

            channels_now += inter_channel

        assert channels_now == concat_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x_nl = self.non_local(x)
        feature_list = [x_nl,]

        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)

        inputs = torch.cat(feature_list, dim=1)
        fusion_outputs = self.fusion(inputs)

        block_outputs = fusion_outputs + x

        return block_outputs
