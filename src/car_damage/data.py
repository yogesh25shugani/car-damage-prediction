from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split

from collections import Counter

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Reproducibility
RANDOM_STATE = 42


# --------------------------------------------------
# 1. Scan and validate dataset
# --------------------------------------------------

def scan_dataset(dataset_dir):
    """
    Scan the dataset directory and validate images.

    Expected structure:
        Dataset/
            F_Breakage/
            F_Crushed/
            F_Normal/
            R_Breakage/
            R_Crushed/
            R_Normal/
    """

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    class_folders = sorted(
        folder for folder in dataset_dir.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        raise ValueError("No class folders found in dataset.")

    class_names = [folder.name for folder in class_folders]

    records = []
    corrupt_images = []

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(class_names)
    }

    for class_folder in class_folders:

        image_files = [
            file for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_files:

            try:
                with Image.open(image_path) as image:
                    image.verify()

                records.append(
                    (image_path, class_to_idx[class_folder.name])
                )

            except (OSError, ValueError, UnidentifiedImageError):
                corrupt_images.append(image_path)

    if not records:
        raise ValueError("No valid images found in dataset.")

    class_counts = Counter(
        class_names[label]
        for _, label in records
    )

    return records, class_names, class_to_idx, class_counts, corrupt_images

# --------------------------------------------------
# 2. Custom PyTorch Dataset
# --------------------------------------------------

class CarDamageDataset(Dataset):

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, label = self.records[index]

        with Image.open(image_path) as image:
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# --------------------------------------------------
# 3. Image transformations
# --------------------------------------------------

def get_transforms(image_size=224):

    # Training transformation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15,contrast=0.15,saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # Validation and test transformations
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

# --------------------------------------------------
# 4. Stratified train/validation/test split
# --------------------------------------------------

def split_dataset(records, random_state=RANDOM_STATE):

    labels = [label for _, label in records]

    indices = list(range(len(records)))

    # 70% training and 30% temporary
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=labels
    )

    temp_labels = [labels[index] for index in temp_indices]

    # Divide remaining 30% equally
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_labels
    )

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
    test_records = [records[i] for i in test_indices]

    return train_records, val_records, test_records


# --------------------------------------------------
# 5. Create PyTorch DataLoaders
# --------------------------------------------------

def build_dataloaders( dataset_dir,
    image_size=224,
    batch_size=8,
    num_workers=0,
    random_state=RANDOM_STATE):

    records, class_names, class_to_idx, class_counts, corrupt_images = (scan_dataset(dataset_dir))

    train_records, val_records, test_records = split_dataset(
        records,
        random_state=random_state
    )

    train_transform, eval_transform = get_transforms(
        image_size=image_size
    )

    train_dataset = CarDamageDataset(
        train_records,
        transform=train_transform
    )

    val_dataset = CarDamageDataset(
        val_records,
        transform=eval_transform
    )

    test_dataset = CarDamageDataset(
        test_records,
        transform=eval_transform
    )

    generator = torch.Generator()
    generator.manual_seed(random_state)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "class_counts": dict(class_counts),
        "corrupt_images": corrupt_images
    }



