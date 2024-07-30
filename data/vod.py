from data.base_dataset import BaseDataset
import numpy as np
import os


class VodData(BaseDataset):
    def __init__(self, vod_path, is_train, grid_conf, final_hw, org_hw, data_aug_conf, radar_folder="radar",
                 radar_augmentation=False,
                 useRadar=True,
                 useCamera=True):
        super(VodData).__init__()
        # This version focus on the semantic segmentation
        self.vod_path = vod_path
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.radar_augmentation = radar_augmentation
        self.radar_folder = radar_folder
        self.grid_conf = grid_conf
        self.final_h, self.final_w = final_hw
        self.org_h, self.org_w = org_hw
        self.useRadar = useRadar
        self.useCamera = useCamera

        self.frame_numbers = self.get_frame_numbers()
        self._setup_directories()
        self._setup_grid_parameters()

    def _setup_directories(self):
        self.radar_calib_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/calib")
        self.cam_data_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/image_2")
        self.radar_data_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/velodyne")
        self.ann_data_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/label_2")

    def _setup_grid_parameters(self):
        '''
        x: left/ right y: up /down z: backward / forward
        nx: The bev grid number in each dim
        dx: The resolution of each grid
        bev_offset: used to convert the points from camera coordinate to BEV image
        bev_scale: The resolution of each grid (ONLY ON BEV IMAGE)
        '''
        self.nx = np.array([(row[1] - row[0]) / row[2] for row in [
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']
        ]]).astype(np.int32)
        self.dx = np.array([row[2] for row in [
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']
        ]])
        self.bx = np.array([row[0] + row[2] / 2.0 for row in [
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']
        ]])
        self.bev_offset = np.array([self.bx[0] - self.dx[0] / 2, self.bx[2] - self.dx[2] / 2])
        self.bev_scale = np.array([self.dx[0], self.dx[2]])

    def get_frame_numbers(self):
        '''
        Get all frame numbers from the val or train txt file
        '''
        frame_num_txt_path = os.path.join(
            self.vod_path, f"{self.radar_folder}/ImageSets/{'train' if self.is_train else 'val'}.txt"
        )
        with open(frame_num_txt_path, 'r') as f:
            return [x.strip() for x in f.readlines()]
