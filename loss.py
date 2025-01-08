# Contains the contrastive loss to use for training
import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467314&isnumber=31472
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        square_pred = torch.pow(euclidean_distance, 2)
        margin_square = torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        loss = label * square_pred + (1 - label) * margin_square
        loss = torch.mean(loss)
        return loss
