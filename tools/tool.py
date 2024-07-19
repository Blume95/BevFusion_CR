import torch
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


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


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_val_info(model, val_loader, loss_fn, device, use_tqdm=True):
    model.eval()
    print('running eval...')
    total_intersect = 0
    total_union = 0
    total_loss = 0.0
    loader = tqdm(val_loader) if use_tqdm else val_loader
    with torch.no_grad():
        for batch in loader:
            img_, extrinsic_, intrinsic_, post_rot_, post_tran_, gt_binimg_, radar_bev = batch
            preds = model(img_.to(device), intrinsic_.to(device), post_rot_.to(device), post_tran_.to(device),
                          radar_bev.to(device))
            gt_binimg_ = gt_binimg_.to(device)

            # loss
            total_loss += loss_fn(preds, gt_binimg_)

            # foreground iou
            pred = (preds > 0)
            tgt = gt_binimg_.bool()
            total_intersect += (pred & tgt).sum().float().item()
            total_union += (pred | tgt).sum().float().item()
        return {
            'loss': total_loss / len(val_loader.dataset),
            'iou': total_intersect / total_union,
        }


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(world_size)
