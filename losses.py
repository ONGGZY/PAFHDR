import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[1, 2, 1],
                    [0, 0, 0],
                    [-1,-2,-1]]
        kernel_h = [[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = kernel_h
        self.weight_v = kernel_v

    def forward(self, x):
        x0 = x[:, 0]  # b,c,h,w
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        grad_magnitude = torch.cat([x0, x1, x2], dim=1)

        grad_h = torch.cat([x0_h, x1_h, x2_h], dim=1)
        grad_v = torch.cat([x0_v, x1_v, x2_v], dim=1)

        return grad_magnitude, grad_h, grad_v


class GradientCosineLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(GradientCosineLoss, self).__init__()
        self.eps = eps

    def forward(self, grad1_h, grad1_v, grad2_h, grad2_v):
        """
        计算两个梯度方向的一致性损失
        Args:
            grad1_h, grad1_v: 图像1的水平和垂直梯度
            grad2_h, grad2_v: 图像2的水平和垂直梯度

        Returns:
            损失值：梯度方向的一致性损失
        """
        # 向量化梯度
        grad1_vector = torch.stack((grad1_h, grad1_v), dim=-1)  # [B, C, H, W, 2]
        grad2_vector = torch.stack((grad2_h, grad2_v), dim=-1)  # [B, C, H, W, 2]

        # 计算余弦相似度
        dot_product = (grad1_vector * grad2_vector).sum(dim=-1)  # [B, C, H, W]
        norm1 = torch.norm(grad1_vector, dim=-1)  # [B, C, H, W]
        norm2 = torch.norm(grad2_vector, dim=-1)  # [B, C, H, W]
        cosine_similarity = dot_product / (norm1 * norm2 + self.eps)

        # 损失：1 - 平均余弦相似度
        loss = 1 - cosine_similarity.mean()
        return loss