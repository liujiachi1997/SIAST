import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

    
class DeconfusionMarginTrainingLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, acsl_thre=0.5, lambda_ = 1):
        super(DeconfusionMarginTrainingLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.acsl_thre = acsl_thre
        self.lambda_ = lambda_

    def forward(self, x, target):
        # self.m_list = self.m_list.to(x.device)
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor).to(x.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        ldam_loss = F.cross_entropy(self.s*output, target, weight=self.weight)

        acsl_logits = torch.sigmoid(output)
        acsl_weight_mask = torch.zeros_like(acsl_logits, dtype=torch.uint8)
        acsl_weight_mask[acsl_logits > self.acsl_thre] = 1
        acsl_weight_mask.scatter_(1, target.data.view(-1, 1), 0)

        num = sum(sum(acsl_weight_mask))
        if num == 0:
            acsl_loss = 0
        else:
            acsl_loss = torch.sum(acsl_weight_mask * F.binary_cross_entropy(acsl_logits, index_float, reduction='none')) / num

        return ldam_loss + self.lambda_ * acsl_loss



if __name__ == "__main__":
    # Example of target with class indices

    loss = LDAMACSLoss(cls_num_list=[20, 10, 100, 10000])
    input = torch.randn(3, 4, requires_grad=True).cuda()
    target = torch.empty(3, dtype=torch.long).random_(4).cuda()
    output = loss(input, target)
    print(input)
    print(target)
    print(output)

    