import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from model.up_conv import Up


class Encoder(nn.Module):
    def __init__(self, out_channel):
        super(Encoder, self).__init__()
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channels=self.trunk._blocks[-1]._bn2.num_features, out_channels=self.out_channel,
                               kernel_size=3,
                               stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3,
                               stride=1, padding=1)

    def process_data(self, x):
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))

        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        return x

    def forward(self, x):
        x = self.process_data(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


class Upper(nn.Module):
    def __init__(self, in_channels, out_channels, up_times):
        super(Upper, self).__init__()
        # channel num [128,]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_times = up_times
        self.modelList = nn.ModuleList()
        self.drop_connect_rate = 0.1
        self.scale_factor_block = 2
        self.build_up()

    def build_up(self, ):
        for i in range(self.up_times):
            if i == 0:
                self.modelList.append(Upper.block(self.in_channels, self.out_channels,
                                                  drop_connect_rate=self.drop_connect_rate,
                                                  scale_factor=self.scale_factor_block))
            else:
                self.modelList.append(Upper.block(self.out_channels, self.out_channels,
                                                  drop_connect_rate=self.drop_connect_rate,
                                                  scale_factor=self.scale_factor_block
                                                  ))

    @staticmethod
    def block(in_channels, out_channels, drop_connect_rate, scale_factor):
        conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        batchnorm0 = nn.BatchNorm2d(out_channels)
        relu0 = nn.ReLU(inplace=True)
        up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        batchnorm1 = nn.BatchNorm2d(out_channels)
        conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        dropout = nn.Dropout(p=drop_connect_rate)

        return nn.Sequential(conv0, batchnorm0, relu0, up, conv1, batchnorm1, dropout, conv2, dropout)

    def forward(self, x):
        for block in self.modelList:
            x = block(x)
        return x


class BevDecode(nn.Module):
    def __init__(self, inC: int, outC: int):
        super(BevDecode, self).__init__()
        trunk = resnet18(weights=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        x = self.up2(x)
        return x


class CVT(nn.Module):
    def __init__(self, input_dim: list, dim: list):
        super(CVT, self).__init__()
        self.out_channels = int(dim[0] * dim[1])
        self.shape_transform = nn.Sequential(
            nn.Linear(int(input_dim[0] * input_dim[1]), self.out_channels),
            nn.ReLU()
        )
        self.forward_transform = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU()
        )
        self.backward_transform = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU()
        )
        self.dim = [int(dim[0]), int(dim[1])]
        pass

    def forward(self, x):
        # x: B, C ,H, W
        x = x.view(x.size(0), x.size(1), -1)
        x = self.shape_transform(x)
        assert x.shape[-1] == self.out_channels
        x_forward = self.forward_transform(x)
        x_backward = self.backward_transform(x_forward)

        x_forward = x_forward.view(x.size(0), x.size(1), self.dim[0], self.dim[1])
        x_backward = x_backward.view(x.size(0), x.size(1), self.dim[0], self.dim[1])
        x = x.view(x.size(0), x.size(1), self.dim[0], self.dim[1])

        return x, x_forward, x_backward


class CrossViewTransformer(nn.Module):
    def __init__(self, in_dim):
        super(CrossViewTransformer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.f_conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.in_dim = in_dim

    def forward(self, x, x_forward, x_backward):
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        query = self.query_conv(x_forward)
        key = self.key_conv(x)
        value = self.value_conv(x_backward)  # B C H W
        value = value.view(B, C, -1)
        # B N C and B C N
        att_score = torch.bmm(key.view(B, self.in_dim // 8, -1).permute(0, 2, 1),
                              query.view(B, self.in_dim // 8, -1))  # B N N
        max_value, idx = att_score.max(dim=-1)
        idx = idx.unsqueeze(1).repeat(1, C, 1)

        selected_value = torch.gather(value, 2, idx).view(B, C, H, W)
        max_value = max_value.view(B, 1, H, W)

        front_res = torch.cat((x, selected_value), dim=1)
        front_res = self.f_conv(front_res)
        front_res = front_res * max_value
        out = x + front_res

        return out


class PYVAModel(nn.Module):
    def __init__(self, feature_size: list, bev_size: list, encoder_chn=128, upper_chn=64, out_chn=1, radar_chn=0,
                 up_times=1):
        super(PYVAModel, self).__init__()
        self.encoder = Encoder(out_channel=encoder_chn)

        self.cvt = CVT(input_dim=[feature_size[0], feature_size[1]], dim=[bev_size[0], bev_size[1]])
        self.attention = CrossViewTransformer(in_dim=encoder_chn)
        self.upper = Upper(encoder_chn, upper_chn, up_times=up_times)
        self.decoder = BevDecode(inC=upper_chn + radar_chn, outC=out_chn)

    def forward(self, x, radar=None):
        x = self.encoder(x)
        x, x_forward, x_backward = self.cvt(x)
        x_org = x
        x = self.attention(x, x_forward, x_backward)
        x = self.upper(x)
        x_forward = self.upper(x_forward)
        if radar is not None:
            x = torch.cat((x, radar), dim=1)
            x_forward = torch.cat((x_forward, radar), dim=1)
        x = self.decoder(x)
        x_forward = self.decoder(x_forward)

        return x, x_org, x_backward, x_forward


# class Discriminator(nn.Module):
#     def __init__(self, input_dim):
#         super(Discriminator, self).__init__()


if __name__ == "__main__":
    pass
