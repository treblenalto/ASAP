import torch
import torch.nn as nn

class JointsRMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsRMSELoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        target_coord = target[:, :, :2]
        target_weight = target[:, :, 2].unsqueeze(-1)

        loss = self.criterion(pred, target_coord)
        if self.use_target_weight:
          loss *= target_weight
          
        loss = torch.sqrt(torch.mean(torch.mean(loss, dim=0)))
        return loss


class OffsetMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(OffsetMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss_hm = 0
        loss_offset = 0
        num_joints = output.size(1) // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx*3].squeeze()
            heatmap_gt = heatmaps_gt[idx*3].squeeze()
            offset_x_pred =  heatmaps_pred[idx*3+1].squeeze()
            offset_x_gt =  heatmaps_gt[idx*3+1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()

            if self.use_target_weight:
                loss_hm += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_x_pred,
                    heatmap_gt * offset_x_gt
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_y_pred,
                    heatmap_gt * offset_y_gt
                )

        return loss_hm / num_joints, loss_offset/num_joints


class OffsetL1Loss(nn.Module):
    def __init__(self, use_target_weight,reduction = 'mean'):
        super(OffsetL1Loss, self).__init__()
        self.criterion = nn.SmoothL1Loss(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.reduction = reduction

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss_hm = 0
        loss_offset = 0
        num_joints = output.size(1) // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx*3].squeeze()
            heatmap_gt = heatmaps_gt[idx*3].squeeze()
            offset_x_pred =  heatmaps_pred[idx*3+1].squeeze()
            offset_x_gt =  heatmaps_gt[idx*3+1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()


            if self.use_target_weight:
                loss_hm += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_x_pred,
                    heatmap_gt * offset_x_gt
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_y_pred,
                    heatmap_gt * offset_y_gt
                )
        if self.reduction == 'mean':
            return loss_hm / num_joints, loss_offset/num_joints
        else:
            return loss_hm,loss_offset

class HeatmapMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class HeatmapOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight=True, topk=8):
        super(HeatmapOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.sum(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)