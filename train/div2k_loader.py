import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")
        return self.transform(lr), self.transform(hr)