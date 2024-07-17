from torch.utils.data import Dataset

class VodData(Dataset):
    def __init__(self, vod_path, is_train, grid_conf, final_hw, org_hw, data_aug_conf, radar_folder="radar"):
        