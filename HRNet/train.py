import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from seed import seed_everything
from model import model_define
from data import AnimalKeypoint
from eval import calc_dists, dist_acc, accuracy
from inference import get_final_preds, get_max_preds
from config import SingleModelConfig
from loss import JointsRMSELoss, OffsetMSELoss, OffsetL1Loss, HeatmapMSELoss, HeatmapOHKMMSELoss

def calc_coord_loss(pred, gt):
    batch_size = gt.size(0)
    valid_mask = gt[:, :, -1].view(batch_size, -1, 1)
    gt = gt[:, :, :2]
    return torch.mean(torch.sum(torch.abs(pred-gt) * valid_mask, dim=-1))

def train(cfg, meta_info_dir, train_img_path, train_tfms=None, valid_tfms=None):
  # for reporduction
  seed = cfg.seed
  torch.cuda.empty_cache()
  seed_everything(2021)

  # device type
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # define model
  if cfg.target_type=='offset':
    pass
  elif cfg.target_type=='gaussian':
    yaml_name = "./config/heatmap_train.yaml"

  yaml_path = os.path.join(cfg.main_dir, yaml_name)
  model = model_define(yaml_path, cfg.init_training)
  model = model.to(device)

  # define criterions
  if cfg.target_type == "offset":
    main_criterion = OffsetMSELoss(True)
  elif cfg.target_type == "gaussian":
    if cfg.loss_type == "MSE":
      main_criterion = HeatmapMSELoss(True)
    elif cfg.loss_type == "OHKMMSE":
      main_criterion = HeatmapOHKMMSELoss(True)
  rmse_criterion = JointsRMSELoss()

  # define optimizer and scheduler
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

  total_df = pd.read_csv(meta_info_dir)

  train_df, valid_df = train_test_split(total_df.iloc[:, :], test_size=cfg.test_ratio, random_state=seed)

  train_ds = AnimalKeypoint(cfg, train_img_path, train_df, train_tfms, mode='train')
  valid_ds  = AnimalKeypoint(cfg, train_img_path, valid_df, valid_tfms, mode='valid')
  train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
  valid_dl  = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)

  print("Train Transformation:\n", train_tfms, "\n")
  print("Valid Transformation:\n", valid_tfms, "\n")


  best_loss = float('INF')
  for epoch in range(cfg.epochs):
      ################
      #    Train     #
      ################
      with tqdm(train_dl, total=train_dl.__len__(), unit="batch") as train_bar:
          train_acc_list = []
          train_rmse_list = []
          train_heatmap_list = []
          train_coord_list = []
          train_offset_list = []
          train_total_list = []

          for sample in train_bar:
              train_bar.set_description(f"Train Epoch {epoch+1}")

              optimizer.zero_grad()
              images, targ_coords = sample['image'].to(device), sample['keypoints'].to(device)
              target, target_weight = sample['target'].to(device), sample['target_weight'].to(device)

              model.train()
              with torch.set_grad_enabled(True):
                  preds = model(images)
                  if cfg.target_type == "offset":
                    loss_hm, loss_os = main_criterion(preds, target, target_weight)
                    loss = loss_hm + loss_os
                  elif cfg.target_type == "gaussian":
                    loss = main_criterion(preds, target, target_weight)

                  if cfg.target_type=="offset":
                    heatmap_height = preds.shape[2]
                    heatmap_width  = preds.shape[3]
                    pred_coords    = get_final_preds(cfg, preds.detach().cpu().numpy())
                  elif cfg.target_type=='gaussian':
                    heatmap_height = preds.shape[2]
                    heatmap_width = preds.shape[3]
                    pred_coords, _ = get_max_preds(preds.detach().cpu().numpy())
                    pred_coords[:, :, 0] = pred_coords[:, :, 0] / (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
                    pred_coords[:, :, 1] = pred_coords[:, :, 1] / (heatmap_height - 1.0) * (4 * heatmap_height - 1.0)

                  pred_coords = torch.tensor(pred_coords).float().to(device)
                  coord_loss  = calc_coord_loss(pred_coords, targ_coords)

                  rmse_loss = rmse_criterion(pred_coords, targ_coords)
                  _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                                                   target.detach().cpu().numpy()[:, ::3, :, :])
                  
                  loss.backward()
                  optimizer.step()

                  if cfg.target_type == "offset":
                    train_heatmap_list.append(loss_hm.item())
                    train_offset_list.append(loss_os.item())
                  train_rmse_list.append(rmse_loss.item())
                  train_total_list.append(loss.item())
                  train_coord_list.append(coord_loss.item())
                  train_acc_list.append(avg_acc)
              train_acc = np.mean(train_acc_list)
              train_rmse = np.mean(train_rmse_list)
              train_coord = np.mean(train_coord_list)
              train_total = np.mean(train_total_list)

              if cfg.target_type == "offset":
                train_offset = np.mean(train_offset_list)
                train_heatmap = np.mean(train_heatmap_list)  
                train_bar.set_postfix(heatmap_loss = train_heatmap,
                                      coord_loss = train_coord,
                                      offset_loss = train_offset,
                                      rmse_loss = train_rmse,
                                      total_loss = train_total,
                                      train_acc  = train_acc)
              else:
                train_bar.set_postfix(coord_loss = train_coord,
                                      rmse_loss = train_rmse,
                                      total_loss = train_total,
                                      train_acc  = train_acc)
      
      ################
      #    Valid     #
      ################
      with tqdm(valid_dl, total=valid_dl.__len__(), unit="batch") as valid_bar:
          valid_acc_list = []
          valid_rmse_list = []
          valid_heatmap_list = []
          valid_coord_list = []
          valid_offset_list = []
          valid_total_list = []
          for sample in valid_bar:
              valid_bar.set_description(f"Valid Epoch {epoch+1}")

              images, targ_coords = sample['image'].to(device), sample['keypoints'].to(device)
              target, target_weight = sample['target'].to(device), sample['target_weight'].to(device)

              model.eval()
              with torch.no_grad():
                  preds = model(images)
                  if cfg.target_type == "offset":
                    loss_hm, loss_os = main_criterion(preds, target, target_weight)
                    loss = loss_hm + loss_os
                  elif cfg.target_type == "gaussian":
                    loss = main_criterion(preds, target, target_weight)
                  
                  pred_coords = get_final_preds(cfg, preds.detach().cpu().numpy())
                  pred_coords = torch.tensor(pred_coords).float().to(device)
                  coord_loss  = calc_coord_loss(pred_coords, targ_coords)

                  rmse_loss = rmse_criterion(pred_coords, targ_coords)
                  _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                                                   target.detach().cpu().numpy()[:, ::3, :, :])
                  
                  if cfg.target_type == "offset":
                    valid_heatmap_list.append(loss_hm.item())
                    valid_offset_list.append(loss_os.item())
                  valid_rmse_list.append(rmse_loss.item())
                  valid_total_list.append(loss.item())
                  valid_coord_list.append(coord_loss.item())
                  valid_acc_list.append(avg_acc)
              valid_acc = np.mean(valid_acc_list)
              valid_rmse = np.mean(valid_rmse_list)
              valid_coord = np.mean(valid_coord_list)
              valid_total = np.mean(valid_total_list)
              if cfg.target_type == "offset":
                valid_offset = np.mean(valid_offset_list)
                valid_heatmap = np.mean(valid_heatmap_list)  
                valid_bar.set_postfix(heatmap_loss = valid_heatmap,
                                      coord_loss = valid_coord,
                                      offset_loss = valid_offset,
                                      rmse_loss = valid_rmse,
                                      total_loss = valid_total,
                                      valid_acc  = valid_acc)
              else:
                valid_bar.set_postfix(coord_loss = valid_coord,
                                      rmse_loss = valid_rmse,
                                      total_loss = valid_total,
                                      valid_acc  = valid_acc)

      if best_loss > valid_total:
          best_model = model
          save_dir =  cfg.save_folder
          save_name = f'best_model_{valid_total}.pth'
          torch.save(model.state_dict(), os.path.join(save_dir, save_name))
          print(f"Valid Loss: {valid_total:.8f}\nBest Model saved.")
          best_loss = valid_total

  return best_model

def main():
    meta_info_dir = '../data/annotations_1.csv'
    train_img_path = '../images/images_1'

    train_tfms = A.Compose([
            A.OneOf([
                A.ChannelShuffle(p=1.0),
                A.HueSaturationValue(p=1.0),
                A.RGBShift(p=1.0),
            ], p=0.5),

            A.RandomBrightnessContrast(p=0.6),
            A.RandomContrast(p=0.6),
            A.RandomGamma(p=0.6),
            A.CLAHE(p=0.5),

            A.Normalize(p=1.0),
        ])

    valid_tfms = A.Normalize(p=1.0)


    cfg = SingleModelConfig(
        epochs=30,
        input_size=[640, 480],   
        learning_rate=5e-4,
        sigma=3.0,
        batch_size=4,  
        shift = True,
        init_training=True, 
        loss_type = "MSE",
        target_type = "gaussian",
        save_folder='./weight'
        )

    best_model = train(cfg, meta_info_dir, train_img_path, train_tfms=train_tfms, valid_tfms=valid_tfms)

if __name__=="__main__":
    main()