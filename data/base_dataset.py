from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
import logging
import torchvision.transforms as T
import cv2


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def get_rot(h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            assert crop_h >= 0 and crop_w >= 0, f"crop_h:{crop_h} or crop_w:{crop_w} is small than zero"
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            assert crop_h >= 0 and crop_w >= 0, f"crop_h:{crop_h} or crop_w:{crop_w} is small than zero"
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    @staticmethod
    def img_transform(img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = BaseDataset.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def get_sensor_transforms(self, sensor, frame_number):
        if sensor == 'radar':
            calibration_file = os.path.join(self.radar_calib_dir, f'{frame_number}.txt')
        elif sensor == 'lidar':
            calibration_file = os.path.join(self.lidar_calib_dir, f'{frame_number}.txt')
        else:
            raise AttributeError('Not a valid sensor')

        try:
            with open(calibration_file, "r") as f:
                lines = f.readlines()
                intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
                extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
                extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
            return intrinsic, extrinsic
        except FileNotFoundError:
            logging.error(f"{frame_number}.txt does not exist at location: {calibration_file}!")
            return None, None

    @staticmethod
    def create_bev_grid(x_range, y_range, z_range):
        """Create a bird's eye view grid based on the given ranges."""
        return np.zeros(
            (
                int((z_range[1] - z_range[0]) / z_range[2]),
                int((x_range[1] - x_range[0]) / x_range[2]),
                int((y_range[1] - y_range[0]) / y_range[2])
            ), dtype=np.float32
        )

    @staticmethod
    def voxelize_pointcloud(num_features, xbound, ybound, zbound, points):
        """Convert a point cloud into a voxel grid."""
        bev_grid_features = [BaseDataset.create_bev_grid(xbound, ybound, zbound) for _ in range(num_features - 3)]

        index = ((points[:, :3] - [xbound[0], ybound[0], zbound[0]]) / [xbound[-1], ybound[-1], zbound[-1]]).astype(int)
        kept = (
                (index[:, 0] >= 0) & (index[:, 0] < bev_grid_features[0].shape[1]) &
                (index[:, 1] >= 0) & (index[:, 1] < bev_grid_features[0].shape[2]) &
                (index[:, 2] >= 0) & (index[:, 2] < bev_grid_features[0].shape[0])
        )

        index = index[kept]
        points = points[kept]

        average_grid_features = []

        for bev_num in range(num_features - 3):
            np.add.at(bev_grid_features[bev_num], (index[:, 2], index[:, 0], index[:, 1]), points[:, 3 + bev_num])

            count_grid = np.zeros_like(bev_grid_features[bev_num])
            np.add.at(count_grid, (index[:, 2], index[:, 0], index[:, 1]), 1)
            count_grid[count_grid == 0] = 1

            average_grid = bev_grid_features[bev_num] / count_grid
            average_grid_features.append(average_grid)

        return np.concatenate(average_grid_features, axis=2)

    @staticmethod
    # TODO: make the radar augmentation works
    def radar_augmentation(radar_features, threshold=0.8):
        H, W, C = radar_features.shape
        for c in range(C):
            max_v = np.max(radar_features[:, :, c])
            min_v = np.min(radar_features[:, :, c])
            rand_v = np.random.random((H, W))
            mask = rand_v <= threshold
            noise_v = np.ones_like(rand_v) * (max_v - min_v)
            noise_v[mask] = 0.0
            radar_features[:, :, c] += noise_v

        return radar_features

    @staticmethod
    def get_info_from_annotation(ann_data_dir, frame_num):
        annotation_file = os.path.join(ann_data_dir, f"{frame_num}.txt")
        with open(annotation_file, 'r') as f:
            return f.readlines()

    @staticmethod
    def get_rotated_box_points(rot, points, center):
        x, y = points
        xc, yc = center
        tem_x, tem_y = x - xc, y - yc
        rot_x = tem_x * np.cos(rot) - tem_y * np.sin(rot)
        rot_y = tem_x * np.sin(rot) + tem_y * np.cos(rot)
        return rot_x + xc, rot_y + yc

    def get_img_data(self, frame_num):
        img_name = os.path.join(self.cam_data_dir, f"{frame_num}.jpg")
        img = Image.open(img_name)
        intrinsic, extrinsic = self.get_sensor_transforms('radar', frame_num)
        intrinsic_rot = intrinsic[:3, :3]

        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

        if self.data_aug_conf is not None:
            img, post_rot2, post_tran2 = BaseDataset.img_transform(img, post_rot, post_tran, resize=resize,
                                                                   resize_dims=resize_dims,
                                                                   crop=crop, flip=flip,
                                                                   rotate=rotate)
        else:
            img = img.resize(resize_dims)
            post_rot *= resize
            post_tran2 = post_tran
            post_rot2 = post_rot


        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        normalize_img = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = normalize_img(img)

        return img, extrinsic, intrinsic_rot, post_rot, post_tran, img_name

    def get_radar_data(self, frame_num):
        radar_name = os.path.join(self.radar_data_dir, f"{frame_num}.bin")
        _, extrinsic = self.get_sensor_transforms('radar', frame_num)
        scan = np.fromfile(radar_name, dtype=np.float32).reshape(-1, 7)
        points, features = scan[:, :3], scan[:, 3:6]
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        # points = np.dot(extrinsic, points.T).T
        points = extrinsic.dot(points.T).T
        points = points[:, :3] / points[:, 3].reshape(-1, 1)

        input_scan = np.concatenate([points, features], axis=1)
        radar_features = BaseDataset.voxelize_pointcloud(
            input_scan.shape[1], self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'],
            input_scan
        )
        if self.is_train and self.radar_augmentation:
            radar_features = self.radar_augmentation(radar_features)
        return radar_features

    def generate_binimg(self, frame_num):
        img = np.zeros((self.nx[2], self.nx[0]))
        lines = self.get_info_from_annotation(self.ann_data_dir, frame_num)

        for line in lines:
            line_list = line.strip().split(' ')
            class_name, dim, loc, rotation_y = line_list[0], line_list[-8:-5], line_list[-5:-2], float(line_list[-2])
            bev_center = (float(loc[0]), float(loc[2]))  # x, y

            bev_top_left = (float(loc[0]) + float(dim[1]) / 2, float(loc[2]) + float(dim[2]) / 2)
            bev_bot_right = (float(loc[0]) - float(dim[1]) / 2, float(loc[2]) - float(dim[2]) / 2)
            bev_top_right = (float(loc[0]) - float(dim[1]) / 2, float(loc[2]) + float(dim[2]) / 2)
            bev_bot_left = (float(loc[0]) + float(dim[1]) / 2, float(loc[2]) - float(dim[2]) / 2)

            bev_top_left = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_top_left, bev_center)
            bev_bot_right = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_bot_right, bev_center)
            bev_top_right = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_top_right, bev_center)
            bev_bot_left = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_bot_left, bev_center)

            bev_top_left = np.round((bev_top_left - self.bev_offset) / self.bev_scale).astype(np.int32)
            bev_bot_right = np.round((bev_bot_right - self.bev_offset) / self.bev_scale).astype(np.int32)
            bev_top_right = np.round((bev_top_right - self.bev_offset) / self.bev_scale).astype(np.int32)
            bev_bot_left = np.round((bev_bot_left - self.bev_offset) / self.bev_scale).astype(np.int32)

            pts = np.array([bev_top_left, bev_top_right, bev_bot_right, bev_bot_left], dtype=np.int32)
            img = cv2.fillPoly(img, pts=[pts], color=1)

        return torch.Tensor(img).unsqueeze(0)

    def __getitem__(self, index):
        frame_number = self.frame_numbers[index]
        out_dict = {"image": None,
                    "radar_features": None,
                    "extrinsic": None,
                    "intrinsic": None,
                    "ground_truth": None,
                    "post_trans": None,
                    "post_rot": None,
                    "img_name": None}
        if self.useRadar:
            out_dict['radar_features'] = self.get_radar_data(frame_number)
        if self.useCamera:
            img, extrinsic, intrinsic, post_rot, post_tran, img_name = self.get_img_data(frame_number)
            out_dict['image'] = img
            out_dict['extrinsic'] = extrinsic
            out_dict['intrinsic'] = intrinsic
            out_dict['post_trans'] = post_tran
            out_dict['post_rot'] = post_rot
            out_dict['img_name'] = img_name

        out_dict['ground_truth'] = self.generate_binimg(frame_number)

        return out_dict

    def __len__(self):
        return len(self.frame_numbers)
