from torch.utils.data import Dataset
from pathlib import Path 
from torchvision.io import read_image
import pandas as pd 
import os
imagenet_base_dir =str(Path("../../projectdata/datasets/imagenet/small").absolute() ) 
imagenet_train_path =imagenet_base_dir + '/train' 
imagenet_val_path =imagenet_base_dir + '/val' 
classes_file = '' 
annotation_file = '' 
class CustomDataset(Dataset):
    def __init__(self, annotations_file,classes_file ,  img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, delimiiter=" ").to_dict() # read 
        self.folder_names = pd.read_csv(classes_file ,delimiter=" " ).iloc[: , :2]
        self.folder_names.columns = ['folder_name', 'number']
        self.folder_names = self.folder_names[['number' , 'folder_name']].to_dict() 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
class ImagenetTrainDataset(CustomDataset): 
    def __int__() : 
        pass 