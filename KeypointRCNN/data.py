import pandas as pd
from animal_keypoint import KeypointDataset

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from typing import Tuple 
import albumentations as A
from albumentations.pytorch import ToTensorV2

def collate_fn(batch: torch.Tensor)->Tuple:
    return tuple(zip(*batch))

# Data Transform & Train-Test-Split
def load_data(train_img_path, train_key_path):
    transforms = A.Compose([
        # A.Resize(500, 500, always_apply=True),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ],  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy')
    )

    total_df = pd.read_csv(train_key_path)
    train_key, valid_key = train_test_split(total_df[:10000], test_size = 0.2, random_state = 42)

    trainset = KeypointDataset(train_img_path, train_key, transforms)
    validset = KeypointDataset(train_img_path, valid_key, transforms)
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(validset, batch_size=4, shuffle = False, num_workers = 4, collate_fn = collate_fn)

    return train_loader, valid_loader