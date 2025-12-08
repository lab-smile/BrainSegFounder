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
class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        # device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(args.device)) # temperature to be used in Loss
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(args.device)).float()) # YY mask out the main diagonal, to computer nominators of both positive and negative pairs

    def forward(self, x_i, x_j):
        # 1. apply an L2 normalization to the features, Default: p = 2
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0) # 1 concatenate 2 output views in the batch dimension. 
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) # matrix multiplication to obtain cosine similiarity: [batch_size x views, batch_size x views]
        sim_ij = torch.diag(sim, self.batch_size)        # positive pairs are shifted from the main diagonal by views(=2), you will shift the main diagonal by views to obtain the similarity scores of positive pairs
        sim_ji = torch.diag(sim, -self.batch_size) 
        positives = torch.cat([sim_ij, sim_ji], dim=0) # concat the 2 1D similarity scores in 1D.
        nom = torch.exp(positives / self.temp)         # this the the final nominator
        denom = self.neg_mask * torch.exp(sim / self.temp) # this is to mask out the main diagonal, which is self-similarity all 1.
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size) # Final take logs of nom/sum(all similarity scores) 


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        # self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        # self.recon_loss = torch.nn.L1Loss().cuda()
        # self.contrast_loss = Contrast(args, batch_size).cuda()
        self.rot_loss = torch.nn.CrossEntropyLoss().to(args.device)
        self.recon_loss = torch.nn.L1Loss().to(args.device)
        self.contrast_loss = Contrast(args, batch_size).to(args.device)     
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)
