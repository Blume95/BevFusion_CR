import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import os
from datetime import datetime
from tools.tool import ddp_setup, SimpleLoss, get_val_info
from data.vod import dataloaders
from model.simple_fusion import LiftSplatShoot
from torch.utils.tensorboard import SummaryWriter
import pickle

datetime_now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def train(rank: int,
          world_size: int,
          parameters: dict
          ):
    path = "/home/jing/Downloads/view_of_delft_PUBLIC/"
    grid = parameters["grid"]
    final_hw = parameters["final_hw"]
    org_hw = parameters["org_hw"]
    workers = parameters['workers']
    batch_size = parameters['batch_size']
    data_aug_conf = parameters['data_aug_conf']
    out_channel = parameters['out_channel']
    method = parameters['method']
    cam_channel = parameters['cam_channel']
    radar_channel = parameters['radar_channel']
    lr = parameters['lr']
    weight_decay = parameters['weight_decay']
    nepochs = parameters['nepochs']
    val_step = parameters['val_step']

    ddp_setup(rank, world_size)
    train_loader, val_loader = dataloaders(path, grid, final_hw=final_hw, org_hw=org_hw, nworkers=workers,
                                           batch_size=batch_size, data_aug_conf=data_aug_conf)

    model = LiftSplatShoot(org_fhw=final_hw, org_hw=org_hw, grid_conf=grid, outC=out_channel, camC=cam_channel, radarC=radar_channel,
                           method=method)
    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = rank

    model.train()
    counter = 0
    experiment_name = datetime_now

    loss_fn = SimpleLoss(pos_weight=4).to(rank)

    if rank == 0:
        logdir = './runs'
        weights_dir = f"{logdir}/{experiment_name}"
        os.makedirs(weights_dir, exist_ok=True)

        writer = SummaryWriter()

        with open(f'{weights_dir}/args.pkl', 'wb') as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for epoch in range(nepochs):
        print(f"Epoch: {epoch}")
        # print(epoch)
        total_intersect = 0.0
        total_union = 0.0
        for index_batch, out_dict in enumerate(
                train_loader):
            opt.zero_grad()

            img_ = out_dict['image']
            intrinsic_ = out_dict['intrinsic']
            post_tran_ = out_dict['post_trans']
            post_rot_ = out_dict['post_rot']
            radar_bev = out_dict['radar_features']
            gt_binimg_ = out_dict['ground_truth']

            # print(img_.shape)
            preds = model(img_.to(device), intrinsic_.to(device), post_rot_.to(device), post_tran_.to(device),
                          radar_bev.to(device))
            gt_binimg_ = gt_binimg_.to(device)
            loss = loss_fn(preds, gt_binimg_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            counter += 1

            if counter % 10 == 0 and rank == 0:
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0 and rank == 0:
                pred = (preds > 0)
                tgt = gt_binimg_.bool()
                total_intersect += (pred & tgt).sum().float().item()
                total_union += (pred | tgt).sum().float().item()

                writer.add_scalar('train/iou', total_intersect / total_union, counter)

            if counter % val_step == 0 and rank == 0:
                val_info = get_val_info(model, val_loader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if rank == 0 and counter % val_step == 0:
                mname = os.path.join(weights_dir, "model_{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.module.state_dict(), mname)
                model.train()

    destroy_process_group()


def main():
    final_hw = (128, 256)
    org_hw = (1216, 1936)
    method = 'lss'
    if method == "lss":
        xbound = [-20.0, 20.0, 0.1]
        ybound = [-10.0, 10.0, 20.0]
        zbound = [0, 40.0, 0.1]
        dbound = [3.0, 43.0, 1.0]
    elif method == "bilinear":
        xbound = [-20.0, 20.0, 0.5]
        ybound = [-10.0, 10.0, 0.5]
        zbound = [0.1, 40.1, 0.1]
        dbound = [3.0, 43.0, 1.0]
    grid = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }

    data_aug_conf = {
        'resize_lim': (0.3, 0.36),
        'final_dim': final_hw,
        'rot_lim': (-5.4, 5.4),
        'H': org_hw[0], 'W': org_hw[1],
        'rand_flip': True,
        'bot_pct_lim': (0.0, 0.1)
    }

    world_size = torch.cuda.device_count()

    out_channel = 1
    use_Radar = False
    use_Cam = True
    radar_channel = 0
    cam_channel = 0

    if use_Radar:
        radar_channel = 3
    if use_Cam:
        cam_channel = 64

    training_parameters = {
        "out_channel": out_channel,
        "batch_size": 12,
        "workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "nepochs": 200,
        "val_step": 300,
        "data_aug_conf": data_aug_conf,
        "grid": grid,
        "final_hw": final_hw,
        "org_hw": org_hw,
        "cam_channel": cam_channel,
        "radar_channel": radar_channel,
        "method":method

    }

    mp.spawn(train, args=(
        world_size, training_parameters),
             nprocs=world_size)


if __name__ == '__main__':
    main()
