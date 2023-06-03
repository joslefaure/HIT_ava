import math

import torch
import torch.nn as nn
import torchvision


class ContextMashing(nn.Module):
    def __init__(self, roi_spatial=7, num_classes=60, dropout=0., bias=False,
                 reduce_dim=2304, hidden_dim=2304, downsample='max2x2', depth=2,
                 kernel_size=3, mlp_1x1=False):
        super(ContextMashing, self).__init__()

        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv3d(reduce_dim * 2, hidden_dim, 1, bias=False)

        # down-sampling
        assert downsample in ['none', 'max2x2']
        if downsample == 'none':
            self.downsample = nn.Identity()
        elif downsample == 'max2x2':
            self.downsample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.gap = nn.AdaptiveAvgPool3d(1)

    # data: features, rois, num_rois, roi_ids, sizes_before_padding
    # returns: outputs
    def forward(self, data):

        feats = data['features']

        h, w = feats.shape[3:]

        roi_feats = data["pooled"]
        roi_ids = data['rois']
        high_order_feats = []
        roi_splitted_actors = torch.split(roi_feats, roi_ids)
        for idx in range(feats.shape[0]):  # iterate over mini-batch
            n_rois = roi_ids[idx]

            if n_rois == 0:
                continue

            eff_h, eff_w = h, w
            bg_feats = feats[idx][:, :eff_h, :eff_w].unsqueeze(0)
            bg_feats = bg_feats.repeat((n_rois, 1, 1, 1, 1))
            actor_feats = roi_splitted_actors[idx]

            tiled_actor_feats = actor_feats.expand_as(bg_feats)
            interact_feats = torch.cat([bg_feats, tiled_actor_feats], dim=1)

            interact_feats = self.downsample(interact_feats)
            interact_feats = self.conv1(interact_feats)
            interact_feats = nn.functional.relu(interact_feats)
            interact_feats = self.gap(interact_feats)
            high_order_feats.append(interact_feats)

        high_order_feats = torch.cat(high_order_feats, dim=0).view(
            data['num_rois'], self.hidden_dim, 1, 1, 1)

        return high_order_feats
