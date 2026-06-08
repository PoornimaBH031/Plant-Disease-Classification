# dataset.py

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


# ==========================================================
# TRAIN TRANSFORMS
# ==========================================================

train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(20),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    transforms.RandomAffine(
        degrees=15,
        translate=(0.1,0.1),
        scale=(0.9,1.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# VALIDATION / TEST TRANSFORMS
# ==========================================================

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# DATASET
# ==========================================================

class PlantDiseaseDataset(Dataset):

    def __init__(
            self,
            image_paths,
            labels,
            transform=None):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================================
# AUTOMATIC FOLDER SCAN
# ==========================================================

def load_dataset(root_dir):

    image_paths = []
    labels = []

    class_names = sorted(
        os.listdir(root_dir)
    )

    class_to_idx = {
        cls_name: idx
        for idx, cls_name
        in enumerate(class_names)
    }

    for cls_name in class_names:

        class_folder = os.path.join(
            root_dir,
            cls_name
        )

        if not os.path.isdir(class_folder):
            continue

        for img_name in os.listdir(class_folder):

            if img_name.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif"
                )
            ):

                image_paths.append(
                    os.path.join(
                        class_folder,
                        img_name
                    )
                )

                labels.append(
                    class_to_idx[cls_name]
                )

    return (
        image_paths,
        labels,
        class_names
    )


# ==========================================================
# DATALOADER
# ==========================================================

def create_dataloader(
        image_paths,
        labels,
        batch_size=16,
        shuffle=True,
        transform=None):

    dataset = PlantDiseaseDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return loader


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    dataset_path = "D:/Classification Using Unsupervised Models/train_fruit"

    image_paths, labels, classes = load_dataset(
        dataset_path
    )

    print("Classes:", classes)
    print("Total Images:", len(image_paths))

    loader = create_dataloader(
        image_paths,
        labels,
        batch_size=16,
        transform=train_transform
    )

    images, targets = next(iter(loader))

    print(images.shape)
    print(targets.shape)