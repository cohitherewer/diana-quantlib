from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms


imagenet_base_dir = str(
    Path("../../projectdata/datasets/imagenet/small").absolute()
)
imagenet_train_path = imagenet_base_dir + "/train"
imagenet_val_path = imagenet_base_dir + "/val"
train_annotation_file = "/imec/other/csainfra/nada64/DianaTraining/dianaquantlib/models/imagenet/train_val_map.txt"
val_annotation_file = "/imec/other/csainfra/nada64/DianaTraining/dianaquantlib/models/imagenet/validation_val_map.txt"


class CustomDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        transform=None,
        target_transform=None,
        img_base_dir="",
    ):  # annotation file contains absolute paths
        self.img_dir_labels = pd.read_csv(annotations_file, delimiter=" ")
        self.img_base_dir = img_base_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dir_labels)

    def __getitem__(self, idx):
        img_path = f"{self.img_base_dir}/{self.img_dir_labels.iloc[idx, 0]}"
        image = Image.open(img_path).convert("RGB")
        label = self.img_dir_labels.iloc[idx, 1] - 1
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label


normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class ImagenetTrainDataset(CustomDataset):
    def __init__(self):
        pre_processing_training_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        super().__init__(
            train_annotation_file,
            pre_processing_training_transform,
            img_base_dir=imagenet_train_path,
        )


class ImagenetValidationDataset(CustomDataset):
    def __init__(self):
        pre_processing_validation_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        super().__init__(
            val_annotation_file,
            pre_processing_validation_transform,
            img_base_dir=imagenet_val_path,
        )
