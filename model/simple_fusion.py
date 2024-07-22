import torch
from torch import nn
from torchvision.models.resnet import resnet18
from torch.distributed import init_process_group, destroy_process_group
import os
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.up_conv import Up
from model.camera_encoder import getCamEncoder


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class LiftSplatShoot(nn.Module):
    def __init__(self, org_fhw: tuple, grid_conf: dict, outC: int, sensor_type: str, camC: int, radarC: int,
                 net_name='convnext'):
        super(LiftSplatShoot, self).__init__()
        self.org_fhw = org_fhw
        self.grid_conf = grid_conf
        self.downsample = 16
        self.frustum = self.create_frustum_1camera()
        self.camC = camC
        self.radarC = radarC
        self.sensor_type = sensor_type
        self.D, _, _, _ = self.frustum.shape

        self.camencode = getCamEncoder(self.D, self.camC, net_name=net_name)
        self.bevencode = BevEncode(inC=self.camC + self.radarC, outC=outC)

        self.nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [self.grid_conf['xbound'],
                                                                           self.grid_conf['ybound'],
                                                                           self.grid_conf['zbound']]])
        self.dx = torch.Tensor([row[2] for row in [self.grid_conf['xbound'],
                                                   self.grid_conf['ybound'],
                                                   self.grid_conf['zbound']]])
        self.bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [self.grid_conf['xbound'],
                                                                  self.grid_conf['ybound'],
                                                                  self.grid_conf['zbound']]])

        self.dx = nn.Parameter(self.dx, requires_grad=False)
        self.bx = nn.Parameter(self.bx, requires_grad=False)
        self.nx = nn.Parameter(self.nx, requires_grad=False)

        self.use_quickcumsum = True

    def create_frustum_1camera(self) -> nn.Parameter:
        ogfH, ogfW = self.org_fhw
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrins: torch.Tensor, post_rots: torch.Tensor, post_trans: torch.Tensor) -> torch.Tensor:
        B = intrins.shape[0]
        points = self.frustum - post_trans.view(B, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat((points[:, :, :, :, :2] * points[:, :, :, :, 2:3],
                            points[:, :, :, :, 2:3]), 4)
        points = torch.inverse(intrins).view(B, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        return points

    def get_cam_feats(self, x: torch.Tensor) -> torch.Tensor:
        B, C, imH, imW = x.shape
        x = self.camencode(x)
        x = x.view(B, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 2, 3, 4, 1)
        return x

    def voxel_splat(self, look_up_table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        Nprime = B * D * H * W
        x = x.reshape(Nprime, C)

        look_up_table = ((look_up_table - (self.bx - self.dx / 2.)) / self.dx).long()
        look_up_table = look_up_table.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        look_up_table = torch.cat((look_up_table, batch_ix), 1)

        kept = (look_up_table[:, 0] >= 0) & (look_up_table[:, 0] < self.nx[0]) & (look_up_table[:, 1] >= 0) & (
                look_up_table[:, 1] < self.nx[1]) & (look_up_table[:, 2] >= 0) & (look_up_table[:, 2] < self.nx[2])
        x = x[kept]
        look_up_table = look_up_table[kept]

        ranks = look_up_table[:, 0] * (self.nx[1] * self.nx[2] * B) + look_up_table[:, 1] * (
                self.nx[2] * B) + look_up_table[:, 2] * B + look_up_table[:, 3]
        sorts = ranks.argsort()
        x, look_up_table, ranks = x[sorts], look_up_table[sorts], ranks[sorts]

        if not self.use_quickcumsum:
            x, look_up_table = cumsum_trick(x, look_up_table, ranks)
        else:
            x, look_up_table = QuickCumsum.apply(x, look_up_table, ranks)

        final = torch.zeros((B, C, self.nx[1], self.nx[2], self.nx[0]), device=x.device)
        final[look_up_table[:, 3], :, look_up_table[:, 1], look_up_table[:, 2], look_up_table[:, 0]] = x

        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def get_voxels(self, x: torch.Tensor, intrins: torch.Tensor, post_rots: torch.Tensor,
                   post_trans: torch.Tensor) -> torch.Tensor:
        geom = self.get_geometry(intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        x = self.voxel_splat(geom, x)
        return x

    def forward(self, x: torch.Tensor, intrins: torch.Tensor, post_rots: torch.Tensor, post_trans: torch.Tensor,
                radar_bev: torch.Tensor) -> torch.Tensor:
        if self.sensor_type == "fusion":
            x = self.get_voxels(x, intrins, post_rots, post_trans)
            radar_bev = radar_bev.permute(0, 3, 1, 2)
            x = torch.cat((x, radar_bev), 1)
        elif self.sensor_type == "radar":
            x = radar_bev.permute(0, 3, 1, 2)
        elif self.sensor_type == "camera":
            x = self.get_voxels(x, intrins, post_rots, post_trans)
        else:
            raise ValueError(f"Unsupported sensor type: {self.sensor_type}")

        x = self.bevencode(x)
        return x


class BevEncode(nn.Module):
    def __init__(self, inC: int, outC: int):
        super(BevEncode, self).__init__()
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


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
