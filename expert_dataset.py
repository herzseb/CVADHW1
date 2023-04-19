from torch.utils.data import Dataset
import os
import json
from torchvision import transforms
from PIL import Image
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root, transform=False, command=None):
        self.data_root = data_root
        self.measurements_path = os.path.join(self.data_root, "measurements")
        self.rgb_path = os.path.join(self.data_root, "rgb")
        # dataset is devided such that one batch only has the same command
        if command == None:
            self.json_files = os.listdir(self.measurements_path)
            self.json_files.sort()
        elif command == 0:
            with open ('json_files_left', 'rb') as fp:
                self.json_files = pickle.load(fp)
        elif command == 1:
            with open ('json_files_right', 'rb') as fp:
                self.json_files = pickle.load(fp)
        elif command == 2:
            with open ('json_files_straight', 'rb') as fp:
                self.json_files = pickle.load(fp)
        elif command == 3:
            with open ('json_files_lanefollow', 'rb') as fp:
                self.json_files = pickle.load(fp)
                
        self.img_names = (os.listdir(self.rgb_path))
        self.img_names.sort()
        self.convert_tensor = transforms.ToTensor()
        self.transform = transform
        # pretrained resnet has a defined image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
   

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        file_name = self.json_files[index]
        img = Image.open(os.path.join(self.rgb_path, os.path.splitext(file_name)[0]+'.png'))
        img = self.convert_tensor(img)
        img = img[[2,1,0,], 90:400, :]
        if self.transform:
            img = self.preprocess(img)
        
        data = json.load(
            open(os.path.join(self.measurements_path, self.json_files[index])))
        return [img, data]
