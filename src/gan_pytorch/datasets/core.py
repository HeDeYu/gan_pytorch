import cvutils
import pyutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data_root, include_patterns=None, exclude_patterns=None):
        super(SimpleDataset, self).__init__()
        self.data_root = data_root

        self.img_path_list = list(
            pyutils.glob_dir(data_root, include_patterns, exclude_patterns)
        )
        self.num_samples = len(self.img_path_list)

    def __getitem__(self, index):
        img = cvutils.imread(self.img_path_list[index])
        img = cvutils.resize_img(img, dsize_wh=(96, 64))
        img = transforms.ToTensor()(img)
        return img, img

    def __len__(self):
        return self.num_samples
