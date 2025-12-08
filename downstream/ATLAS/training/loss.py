# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F


# YY comment on 7-14-2023
# get ideas from https://theaisummer.com/simclr/
# simCLR loss NT-Xent: (similarity score of positive pairs)/ (sum of similarity score of both positive and negative pairs)
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class Loss(torch.nn.Module):
    def __init__(self, batch_size: int, device: int):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().to(device)
        self.recon_loss = torch.nn.L1Loss().to(device)
        self.contrast_loss = ContrastiveLoss().to(device)
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, predicted_rotations: torch.Tensor, ground_truth_rotations: torch.Tensor,
                 first_contrastive: torch.Tensor, second_contrastive: torch.Tensor,
                 predicted_images: torch.Tensor, ground_truth_images: torch.Tensor):
        print(f'{predicted_rotations=}')
        ground_truth_rotations_ = torch.zeros(4, dtype=torch.int8)
        ground_truth_rotations_[ground_truth_rotations] = 1
        print(f'{ground_truth_rotations=}')
        rot_loss = self.alpha1 * self.rot_loss(predicted_rotations, ground_truth_rotations)
        print(f'{rot_loss=}')
        contrast_loss = self.alpha2 * self.contrast_loss(first_contrastive, second_contrastive)
        print(contrast_loss)
        recon_loss = self.alpha3 * self.recon_loss(predicted_images, ground_truth_images)
        print(recon_loss)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)