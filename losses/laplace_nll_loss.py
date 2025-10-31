# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    # 拉普拉斯负对数似然损失（Laplace Negative Log-Likelihood Loss）
    # 用于计算多模态轨迹预测中"位置+不确定性"的回归损失
    # 核心是基于拉普拉斯分布建模轨迹位置的概率，损失值越小表示预测越接近真实轨迹

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        # 初始化父类（nn.Module）
        super(LaplaceNLLLoss, self).__init__()
        
        # 保存关键参数：控制数值稳定性和损失聚合方式
        self.eps = eps  # 最小尺度阈值（避免scale过小时计算log(2*scale)出现数值错误）
        self.reduction = reduction  # 损失聚合方式（'mean'：均值，'sum'：求和，'none'：不聚合）

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # 前向传播：计算预测值与真实值的拉普拉斯负对数似然损失
        # pred：模型预测的轨迹特征，形状通常为[F, N, H, 4]（F=模态数，N=智能体数，H=未来步，4=位置x/y + 尺度x/y）
        # target：真实轨迹，形状通常为[N, H, 2]（仅含位置x/y，与pred的位置维度对应）
        
        # 1. 拆分预测结果：将pred最后一维拆分为"位置（loc）"和"尺度（scale）"
        # chunk(2, dim=-1)：按最后一维（dim=-1）平均拆分，4→2+2（loc占前2维，scale占后2维）
        loc, scale = pred.chunk(2, dim=-1)
        
        # 2. 尺度参数处理：确保scale不小于eps，避免数值不稳定
        # scale.clone()：深拷贝scale（避免修改原始预测值，不影响后续计算）
        # with torch.no_grad()：禁用梯度计算（仅对scale做数值截断，不参与反向传播）
        # clamp_(min=self.eps)：将scale中小于eps的值强制设为eps（防止log(2*scale)出现负无穷）
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        
        # 3. 计算拉普拉斯负对数似然（NLL）
        # 拉普拉斯分布的概率密度函数（PDF）：f(x; μ, b) = 1/(2b) * exp(-|x-μ|/b)
        # 负对数似然（NLL）：-log(f(x; μ, b)) = log(2b) + |x-μ|/b
        # 其中：μ=loc（预测位置），b=scale（尺度参数，控制分布分散程度），x=target（真实位置）
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        
        # 4. 按指定方式聚合损失
        if self.reduction == 'mean':
            # 均值聚合：计算所有元素的平均损失（适合批量训练，平衡不同样本的权重）
            return nll.mean()
        elif self.reduction == 'sum':
            # 求和聚合：计算所有元素的总损失（适合需要精确累加损失的场景）
            return nll.sum()
        elif self.reduction == 'none':
            # 不聚合：返回原始损失矩阵（适合需要分析单个样本/模态损失的场景）
            return nll
        else:
            # 无效聚合方式：抛出异常，提示合法选项
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
