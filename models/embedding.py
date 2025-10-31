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
from typing import List, Optional

import torch
import torch.nn as nn

from utils import init_weights


class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):
    # 多输入嵌入器：处理多个不同来源的连续特征（如轨迹向量+相对位置向量）
    # 核心功能是将多个独立输入特征分别编码、再聚合为统一维度的嵌入特征

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        # 初始化父类（nn.Module）
        super(MultipleInputEmbedding, self).__init__()
        
        # 1. 创建输入特征的独立编码模块列表（ModuleList）
        # 每个输入特征对应一个独立的编码链：Linear（维度映射）→ LayerNorm（稳定训练）→ ReLU（非线性）→ Linear（特征精炼）
        # in_channels：输入特征的维度列表（如[node_dim, edge_dim]，对应轨迹向量、相对位置向量）
        # 遍历in_channels，为每个输入维度创建一个相同结构的编码模块
        self.module_list = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_channel, out_channel),  # 第一步：将单个输入特征从in_channel映射到目标out_channel
                nn.LayerNorm(out_channel),           # 第二步：层归一化，避免特征值漂移，加速收敛
                nn.ReLU(inplace=True),               # 第三步：ReLU非线性激活，引入特征非线性表达能力
                nn.Linear(out_channel, out_channel)  # 第四步：再次线性变换，精炼特征（避免简单维度映射导致的信息损失）
            ) for in_channel in in_channels]  # 为每个输入维度生成一个编码链
        )
        
        # 2. 创建特征聚合模块（aggr_embed）
        # 对多个编码后的特征求和聚合后，进一步做非线性变换和归一化，得到最终嵌入特征
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),           # 第一步：聚合后先归一化，稳定特征分布
            nn.ReLU(inplace=True),               # 第二步：非线性激活，融合多特征的交互信息
            nn.Linear(out_channel, out_channel), # 第三步：线性精炼，确保聚合后特征维度仍为out_channel
            nn.LayerNorm(out_channel)            # 第四步：再次归一化，为后续模块提供稳定输入
        )
        
        # 应用自定义权重初始化函数（如Xavier初始化），确保各层参数初始状态合理，避免训练发散
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        # forward函数：多输入特征的编码与聚合流程
        # continuous_inputs：连续型输入特征列表（如[轨迹向量张量, 相对位置向量张量]）
        # categorical_inputs：可选的离散型输入特征列表（如智能体类型、车道类型，默认None）
        
        # 1. 逐个编码连续型输入特征
        # 遍历module_list和continuous_inputs，用对应编码链处理每个连续特征
        for i in range(len(self.module_list)):
            # 将第i个连续输入特征传入第i个编码链，更新为编码后的特征
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        
        # 2. 聚合连续型特征：将所有编码后的连续特征按维度0堆叠（形成[输入数量, N, embed_dim]），再求和（得到[N, embed_dim]）
        # 例如：2个输入特征（轨迹+相对位置）→ 堆叠后[2, N, 64] → 求和后[N, 64]，实现特征融合
        output = torch.stack(continuous_inputs).sum(dim=0)
        
        # 3. （可选）加入离散型特征：若存在categorical_inputs，同样堆叠求和后加到聚合结果中
        # 离散特征通常已提前编码为与out_channel匹配的向量（如Embedding层处理）
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        
        # 4. 聚合特征精炼：将融合后的特征传入aggr_embed，得到最终嵌入结果
        return self.aggr_embed(output)
