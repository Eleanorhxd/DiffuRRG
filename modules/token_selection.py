from torch import nn
import torch
from copy import deepcopy as clone
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn

class TokenSelection(nn.Module):
    def __init__(self, fore_t=0.5, back_t=0.3):
        super(TokenSelection, self).__init__()
        self.fore_t = fore_t
        self.back_t = back_t

    def forward(self,patch_feats,head,cls_idx,logits):
        cam = self.compute_scores(patch_feats, head, cls_idx)
        logits = torch.sigmoid(logits)
        labels = (logits >= 0.5).float()
        #print(labels[0])
        cam = labels.unsqueeze(-1) * cam
        dm, _ = torch.max(cam, dim=1, keepdim=True)
        fore_idxs = dm > self.fore_t
        return fore_idxs

    def compute_scores(self,patch_feats, fc_layer, class_idx):
        weights = self._get_weights(fc_layer, class_idx)
        with torch.no_grad():

            cams = torch.matmul(patch_feats, weights.transpose(-2,-1)).transpose(-2,-1)


            if self.relu:
                cams = F.relu(cams, inplace=True)

            if self.normalized:
                cams = self._normalize(cams)


        return cams

    @staticmethod
    @torch.no_grad()
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization."""
        cams.sub_(cams.min(-1).values[(..., None)])
        cams_max = cams.max(-1).values[(..., None)]
        cams_max = torch.clamp(cams_max, min = 1e-12, max = 1)
        cams.div_(cams_max)
        return cams


    @torch.no_grad()
    def _get_weights(self,fc_layer, class_idx):
        fc_weights = fc_layer.weight.data
        if fc_weights.ndim > 2:
            fc_weights = fc_weights.view(*fc_weights.shape[:2])
        if isinstance(class_idx, int):
            return fc_weights[class_idx, :].unsqueeze(0)
        else:
            return fc_weights[class_idx, :]