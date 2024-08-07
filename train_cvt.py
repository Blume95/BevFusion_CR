import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import os
from datetime import datetime
from tools.tool import ddp_setup
from data.vod import dataloaders
from model.PYVA import PYVAModel
from torch.utils.tensorboard import SummaryWriter
import pickle
from tools.loss import computeLoss

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
    out_channel = parameters['out_channel']
    radar_channel = parameters['radar_channel']
    lr = parameters['lr']
    weight_decay = parameters['weight_decay']
    nepochs = parameters['nepochs']
    val_step = parameters['val_step']
    use_Camera = parameters['use_Cam']
    use_radar = parameters['use_Radar']

    bev_size = [int((grid['zbound'][1] - grid['zbound'][0]) / grid['zbound'][2]),
                int((grid['xbound'][1] - grid['xbound'][0]) / grid['xbound'][2])]

    ddp_setup(rank, world_size)

    train_loader, val_loader = dataloaders(path, grid, final_hw=final_hw, org_hw=org_hw, nworkers=workers,
                                           batch_size=batch_size, data_aug_conf=None, useRadar=use_radar,
                                           useCamera=use_Camera)

    model = PYVAModel(feature_size=[int(final_hw[0] / 32), int(final_hw[1] / 32)],
                      bev_size=[bev_size[0] / 8, bev_size[1] / 8], radar_chn=radar_channel)
    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    counter = 0
    experiment_name = datetime_now
    loss_fn = computeLoss().to(rank)

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

            img = out_dict['image'].to(rank)
            radar_bev = out_dict['radar_features']
            gt = out_dict['ground_truth'].to(rank)

            if radar_bev[0] == 0:
                pred, x, x_backward = model(img)
            else:
                pred, x, x_backward = model(img, radar_bev.to(rank))
            loss, loss_cycle, loss_bce = loss_fn(pred, gt, x, x_backward)
            loss.backward()
            opt.step()
            counter += 1

            if counter % 10 == 0 and rank == 0:
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0 and rank == 0:
                pred_mask = (pred > 0)
                tgt = gt.bool()
                total_intersect += (pred_mask & tgt).sum().float().item()
                total_union += (pred_mask | tgt).sum().float().item()

                writer.add_scalar('train/iou', total_intersect / total_union, counter)

            # if counter % val_step == 0 and rank == 0:
            #     val_info = get_val_info(model, val_loader, loss_fn, device)
            #     print('VAL', val_info)
            #     writer.add_scalar('val/loss', val_info['loss'], counter)
            #     writer.add_scalar('val/iou', val_info['iou'], counter)
            #
            # if rank == 0 and counter % val_step == 0:
            #     mname = os.path.join(weights_dir, "model_{}.pt".format(counter))
            #     print('saving', mname)
            #     torch.save(model.module.state_dict(), mname)
            #     model.train()


def main():
    final_hw = (128, 256)
    org_hw = (1216, 1936)

    xbound = [-8.0, 8.0, 0.1]
    ybound = [-10.0, 10.0, 20.0]
    zbound = [0, 32, 0.1]
    dbound = [3.0, 43.0, 1.0]
    grid = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
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
        "batch_size": 2,
        "workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "nepochs": 200,
        "val_step": 300,
        "grid": grid,
        "final_hw": final_hw,
        "org_hw": org_hw,
        "cam_channel": cam_channel,
        "radar_channel": radar_channel,
        "use_Radar": use_Radar,
        "use_Cam": use_Cam

    }

    mp.spawn(train, args=(
        world_size, training_parameters),
             nprocs=world_size)


if __name__ == '__main__':
    main()
