import torch
from data.vod_dataset import dataloaders
from model.simple_fusion import LiftSplatShoot
import numpy as np
import cv2
from tools.tool import ddp_setup
def parameters():
    final_hw = (256, 512)
    org_hw = (1216, 1936)

    xbound = [-20.0, 20.0, 0.1]
    ybound = [-10.0, 10.0, 20.0]
    zbound = [0, 40.0, 0.1]
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

    out_channel = 1
    fusion_type = "fusion"
    if fusion_type == "fusion":
        cam_channel = 64
        radar_channel = 3
    elif fusion_type == 'radar':
        cam_channel = 0
        radar_channel = 3
    elif fusion_type == 'camera':
        cam_channel = 64
        radar_channel = 0
    else:
        raise ValueError(f"Unsupported sensor type: {fusion_type}")

    training_parameters = {
        "out_channel": out_channel,
        "fusion_type": fusion_type,
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
        'net_name': "convnext",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    }
    return training_parameters


def vis(parameters):
    path = "/home/jing/Downloads/view_of_delft_PUBLIC/"
    grid = parameters["grid"]
    final_hw = parameters["final_hw"]
    org_hw = parameters["org_hw"]
    workers = parameters['workers']
    batch_size = parameters['batch_size']
    data_aug_conf = parameters['data_aug_conf']
    out_channel = parameters['out_channel']
    sensor_type = parameters['fusion_type']
    cam_channel = parameters['cam_channel']
    radar_channel = parameters['radar_channel']
    lr = parameters['lr']
    weight_decay = parameters['weight_decay']
    nepochs = parameters['nepochs']
    val_step = parameters['val_step']
    net_name = parameters['net_name']
    device = parameters['device']

    ddp_setup(0, 1)

    train_loader, val_loader = dataloaders(path, grid, final_hw=final_hw, org_hw=org_hw, nworkers=workers,
                                           batch_size=1, data_aug_conf=data_aug_conf)

    model = LiftSplatShoot(org_fhw=final_hw, grid_conf=grid, outC=out_channel, sensor_type=sensor_type,
                           camC=cam_channel, radarC=radar_channel, net_name=net_name)

    model.load_state_dict(
        torch.load("/home/jing/Downloads/bev_result/weights/model_2400.pt", map_location=device))
    model.to(device)
    model.eval()
    total_intersect = 0
    total_union = 0
    with torch.no_grad():
        for index_batch, (img_, extrinsic_, intrinsic_, post_rot_, post_tran_, gt_binimg_, radar_bev, image_name) in enumerate(
                val_loader):
            preds = model(img_.to(device), intrinsic_.to(device), post_rot_.to(device), post_tran_.to(device),
                          radar_bev.to(device),image_name)

            preds_np = preds[0, 0, :, :].cpu()
            preds_np_mask = (preds_np > 0)
            # preds_np_mask_bc = (preds_np <= 0)

            gt_np = gt_binimg_[0, 0, :, :]
            gt_np_mask = (gt_np > 0)
            # gt_np_mask_bc = (gt_np <= 0)

            total_intersect += torch.sum(preds_np_mask & gt_np_mask)
            # total_intersect += torch.sum(preds_np_mask_bc & gt_np_mask_bc)
            total_union += torch.sum(preds_np_mask | gt_np_mask)
            # total_union += torch.sum(preds_np_mask_bc | gt_np_mask_bc)

            print(total_intersect / total_union)

            for i in range(preds.shape[0]):
                show_img = cv2.imread(image_name[i], -1)
                pred_np = preds[i].sigmoid().cpu() > 0.01
                pred_np = (pred_np.numpy() * 255).astype(np.uint8)
                gt_np = gt_binimg_[i].cpu().numpy()
                gt_np = (gt_np * 255).astype(np.uint8)
                tem_pred = np.zeros((3, pred_np.shape[1], pred_np.shape[2]), np.uint8)
                tem_gt = np.zeros((3, pred_np.shape[1], pred_np.shape[2]), np.uint8)

                # print(gt_np.shape)

                # print(show_img)

                tem_gt += gt_np
                tem_pred += pred_np

                tem_gt = np.moveaxis(tem_gt, 0, -1)
                tem_pred = np.moveaxis(tem_pred, 0, -1)
                # print(tem_pred)

                # tem_gt = cv2.putText(tem_gt, "BEV_GT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                #                      2, cv2.LINE_AA)
                # tem_pred = cv2.putText(tem_pred, "BEV_Ped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                #                        2, cv2.LINE_AA)

                # pred_np = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2RGB)
                # gt_np = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2RGB)
                # img_np = img_[i].detach().cpu().numpy()
                mid_ = (np.ones((tem_gt.shape[0], 40, 3), dtype=np.uint8) * [127, 127, 127]).astype(np.uint8)

                bev_view = np.concatenate((tem_gt, mid_, tem_pred), axis=1)
                # print(bev_view.shape)
                bev_view = cv2.resize(bev_view, (1400, 1120), interpolation=cv2.INTER_CUBIC)

                # print(bev_view.shape)
                bev_view = cv2.copyMakeBorder(bev_view, 48, 48, 248, 248, cv2.BORDER_CONSTANT, value=[127, 127, 127])
                # print(bev_view.shape, show_img.shape)
                bev_view = cv2.flip(bev_view, 0)

                show_img = np.concatenate([show_img, bev_view], axis=1)
                # show_img = np.moveaxis(show_img, 0, -1)

                cv2.imwrite("/home/jing/Downloads/bev_result/vis/" + f'{image_name[i].split("/")[-1]}',
                            show_img)


if __name__ == '__main__':
    par = parameters()
    vis(par)