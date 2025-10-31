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
import torch.nn.functional as F


class SoftTargetCrossEntropyLoss(nn.Module):
    # 软目标交叉熵损失（Soft Target Cross-Entropy Loss）
    # 用于多模态轨迹预测中"模态概率权重"的分类损失
    # 核心区别于普通交叉熵：支持"软目标"（目标概率可非0/1，如[0.8, 0.1, 0.1]），而非仅"硬目标"（如[1, 0, 0]）
    # 对应论文中优化多模态混合系数（mixing coefficients）的分类损失部分

    def __init__(self, reduction: str = 'mean') -> None:
        # 初始化父类（nn.Module）
        super(SoftTargetCrossEntropyLoss, self).__init__()
        
        # 保存损失聚合方式：控制最终输出的损失形式
        self.reduction = reduction  # 可选值：'mean'（均值）、'sum'（求和）、'none'（不聚合）

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # 前向传播：计算预测模态概率与软目标概率的交叉熵损失
        # pred：模型预测的模态"对数几率"（logits），形状通常为[N, F]（N=智能体数，F=模态数）
        # target：软目标概率（真实模态分布），形状与pred一致[N, F]，且每一行和为1（如[0.9, 0.1, 0, 0, 0, 0]）
        
        # 1. 计算交叉熵损失：软目标版本
        # 步骤1：F.log_softmax(pred, dim=-1) → 将pred（logits）转为"对数概率"（log-prob）
        #  - dim=-1：按最后一维（模态维度F）计算softmax，确保每个智能体的所有模态对数概率和为0（log(1)）
        # 步骤2：-target * 对数概率 → 逐元素相乘并取负（对应交叉熵的核心公式）
        # 步骤3：torch.sum(..., dim=-1) → 按模态维度（F）求和，得到每个智能体的交叉熵损失（形状[N]）
        # 公式对应：H(p, q) = -∑(p_i * log q_i)，其中p=target（真实分布），q=pred_softmax（预测分布）
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        
        # 2. 按指定方式聚合损失
        if self.reduction == 'mean':
            # 均值聚合：计算所有智能体的平均损失（适合批量训练，平衡不同智能体的权重）
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            # 求和聚合：计算所有智能体的损失总和（适合需要精确累加损失的场景）
            return cross_entropy.sum()
        elif self.reduction == 'none':
            # 不聚合：返回每个智能体的独立损失（适合分析单个智能体的模态预测误差）
            return cross_entropy
        else:
            # 无效聚合方式：抛出异常，提示合法选项
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
