from torch import nn
from efficientnet_pytorch import EfficientNet
from model.up_conv import Up
import torch
import torchvision.models as models


class CamEncodeEffB0(nn.Module):
    def __init__(self, D: int, C: int):
        super(CamEncodeEffB0, self).__init__()
        self.D = D
        self.C = C
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        return x.softmax(dim=1)

    def get_depth_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_eff_depth(x)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return new_x

    def get_eff_depth(self, x: torch.Tensor) -> torch.Tensor:
        endpoints = dict()
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f'reduction_{len(endpoints) + 1}'] = prev_x
            prev_x = x

        endpoints[f'reduction_{len(endpoints) + 1}'] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x: torch.Tensor, simple_bev: bool) -> torch.Tensor:
        # x batch , 3 , h, w
        if simple_bev:
            x = self.get_eff_depth(x)
            x = self.depthnet(x)
        else:
            x = self.get_depth_feat(x)
        return x


class CamEncoderConvNextTiny(nn.Module):
    def __init__(self, D: int, C: int):
        super(CamEncoderConvNextTiny, self).__init__()
        self.D = D
        self.C = C
        self.convnext_tiny = models.convnext_tiny(pretrained=True)
        # downsample 4
        self.part0 = nn.Sequential(
            self.convnext_tiny.features[0],
            self.convnext_tiny.features[1]
        )
        self.part1 = nn.Sequential(
            self.convnext_tiny.features[2],
            self.convnext_tiny.features[3]
        )

        self.part2 = nn.Sequential(
            self.convnext_tiny.features[4],
            self.convnext_tiny.features[5]
        )
        self.part3 = nn.Sequential(
            self.convnext_tiny.features[6],
            self.convnext_tiny.features[7]
        )
        self.up1 = Up(384 + 768, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        return x.softmax(dim=1)

    def get_depth_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_eff_depth(x)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return new_x

    def get_eff_depth(self, x: torch.Tensor) -> torch.Tensor:
        endpoints = dict()
        x = self.part0(x)
        x = self.part1(x)
        x = self.part2(x)
        endpoints["pre_layer"] = x
        x = self.part3(x)
        endpoints["last_layer"] = x
        x = self.up1(endpoints["last_layer"], endpoints["pre_layer"])
        return x

    def forward(self, x: torch.Tensor, simple_bev: bool) -> torch.Tensor:
        # x batch , 3 , h, w
        if simple_bev:
            x = self.get_eff_depth(x)
            x = self.depthnet(x)
        else:
            x = self.get_depth_feat(x)
        return x


def getCamEncoder(D, C, net_name='convnext'):
    if net_name == 'convnext':
        camEncoder = CamEncoderConvNextTiny(D, C)
    elif net_name == 'eff_b0':
        camEncoder = CamEncodeEffB0(D, C)
    else:
        raise ValueError(f"{net_name} No implement")

    return camEncoder


if __name__ == "__main__":
    1
