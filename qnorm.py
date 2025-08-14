# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class QualityNorm(nn.Module):
    def __init__(self, num_features, affine=True, quality_channels=64, hidden_dim=128, track_running_stats=False):
        super(QualityNorm, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.quality_channels = quality_channels
        self.track_running_stats = track_running_stats
        self.affine = affine
        self.eps = 1e-5
        self.momentum = 0.1


        self.quality_reshape = nn.Sequential(
            nn.Linear(4096, hidden_dim),  # [b, 4096] -> [b, hidden_dim]
            nn.ReLU(),
        )

        self.gamma_fc = nn.Linear(hidden_dim, num_features)  # [b, hidden_dim] -> [b, num_features]
        self.beta_fc = nn.Linear(hidden_dim, num_features)   # [b, hidden_dim] -> [b, num_features]

        self.gamma_conv = nn.Conv2d(self.hidden_dim, self.num_features, kernel_size=1)
        self.beta_conv = nn.Conv2d(self.hidden_dim, self.num_features, kernel_size=1)

        # Batch Normalization
        self.batch_norm = nn.BatchNorm2d(self.num_features, affine=False)



    def forward(self, x, quality):
        b, c, h, w = x.size()
        # x=self.batch_norm(x)
        function="spatial"#spatial,channel,spatial_channel,channel_spatial

        weight_dtype = next(self.quality_reshape.parameters()).dtype


        quality = quality.to(dtype=weight_dtype)

        if self.affine:

            quality_feature1 = self.quality_reshape(quality)  # [b, 4096] -> [b, hidden_dim]

            ##spatial
            quality_feature = quality_feature1.view(b, self.hidden_dim, 1, 1)
            gamma_s = self.gamma_conv(quality_feature)  # [b, c, 8, 8]
            beta_s = self.beta_conv(quality_feature)  # [b, c, 8, 8]
            gamma_s = F.interpolate(gamma_s, size=(h, w), mode='bilinear')  # [b, c, h, w]
            beta_s = F.interpolate(beta_s, size=(h, w), mode='bilinear')  # [b, c, h, w]

            ##channel
            gamma = self.gamma_fc(quality_feature1)  # [b, num_features]
            beta = self.beta_fc(quality_feature1)  # [b, num_features]
            gamma_c = gamma.view(b, c, 1, 1)  # [b, c] -> [b, c, 1, 1]
            beta_c = beta.view(b, c, 1, 1)  # [b, c] -> [b, c, 1, 1]

            if function == "spatial":
                out = x * (1 + gamma_s) + beta_s

            elif function == "channel":
                out = x * (1 + gamma_c) + beta_c

            elif function == "spatial_channel":
                out = x * (1 + gamma_s) + beta_s
                out = out * (1 + gamma_c) + beta_c

            elif function == "channel_spatial":
                out = x * (1 + gamma_c) + beta_c
                out = out * (1 + gamma_s) + beta_s

        else:
            out=self.batch_norm(x)

        return out
