import os
import cv2
import time
import pandas as pd
import numpy as np
import natsort
from typing import Tuple, Sequence, Callable, Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2



class KeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_df: pd.DataFrame,
        transforms: Sequence[Callable]=None
    ) -> None:
        self.image_dir = image_dir
        self.df = label_df
        self.transforms = transforms

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_id = self.df.iloc[index, 0]
        labels = np.array([1])
        keypoints = self.df.iloc[index, 1:].values.reshape(-1, 2).astype(np.int64)

        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

        image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.COLOR_BGR2RGB)

        targets ={
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets['image']
        image = image / 255.0

        targets = {
            'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
            'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
            'keypoints': torch.as_tensor(
                np.concatenate([targets['keypoints'], np.ones((15, 1))], axis=1)[np.newaxis], dtype=torch.float32
            )
        }

        return image, targets



# Video -> Frame
def get_frame(vidcap, sec, count, frame_path):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    success, image = vidcap.read()
    if not os.path.exists(frame_path):
      os.makedirs(frame_path)
    if success:  # if there is frame > save as jpg file
      cv2.imwrite(os.path.join(frame_path, 'image_'+str(count)+'.jpg'), image)
    return success

def get_video(video_path, frame_path):
    vidcap = cv2.VideoCapture(video_path)
    sec = 0
    frame_rate = 0.2
    count = 1
    success = get_frame(vidcap, sec, count, frame_path)
    while success:
      count+=1
      sec+= frame_rate
      sec = round(sec, 2)
      success = get_frame(vidcap, sec, count, frame_path)
    return True

# Predict Keypoints
def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))

def pred_keypoints(frame_path, model_path, device = 'cuda'):
  model = torch.load(model_path)
  model.to(device)
  model.eval()

  pred_list = []
  frame_no = len(os.listdir(frame_path))
  frames = natsort.natsorted(os.listdir(frame_path))
  col = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x',
       '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x',
       '9_y', '10_x', '10_y', '11_x', '11_y', '12_x', '12_y', '13_x', '13_y',
       '14_x', '14_y']
  test_df = pd.DataFrame(np.zeros((frame_no, 30)), columns = col)
  test_df.insert(0, 'image', frames)

  transforms = A.Compose([
      # A.Resize(500, 500, always_apply=True),
      A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ToTensorV2()
  ],  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
      keypoint_params=A.KeypointParams(format='xy')
  )

  testset = KeypointDataset(frame_path, test_df[:], transforms)
  test_loader = DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2, collate_fn = collate_fn)

  with torch.no_grad():
    for images, targets in test_loader:
      # data, target 값 DEVICE에 할당
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  

      predictions = model(images)
      for i in range(len(predictions)):
        pred_list.append(predictions[i]['keypoints'].cpu().numpy().copy()[0][:, :3].tolist())

  return pred_list

  