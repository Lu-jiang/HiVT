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
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


class GRUDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):
    # MLP解码器：融合局部编码器的特征（local_embed）和全局交互模块的特征（global_embed）
    # 核心功能是输出多智能体的多模态未来轨迹预测，包含轨迹位置（可选不确定性）和模式概率
    # 对应论文中"Multimodal Future Decoder"部分，基于混合拉普拉斯分布建模多模态轨迹

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        # 初始化父类（nn.Module）
        super(MLPDecoder, self).__init__()
        
        # 保存关键参数：控制输入特征维度、预测步长、模态数量、是否输出不确定性
        self.input_size = global_channels    # 全局特征维度（global_embed的维度）
        self.hidden_size = local_channels    # 局部特征维度（local_embed的维度，同时作为中间层维度）
        self.future_steps = future_steps     # 未来预测时间步数（论文中30步，对应3秒）
        self.num_modes = num_modes           # 多模态数量（论文中6种，覆盖不同可能轨迹）
        self.uncertain = uncertain           # 是否预测轨迹不确定性（True：输出位置+尺度；False：仅输出位置）
        self.min_scale = min_scale           # 不确定性尺度的最小值（避免尺度过小导致数值不稳定）

        # 1. 特征聚合嵌入层：融合局部特征和全局特征
        # 输入：局部特征（local_embed）+ 全局特征（global_embed），维度为(input_size + hidden_size)
        # 输出：统一维度的融合特征（hidden_size），通过LayerNorm和ReLU确保特征稳定
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),  # 维度融合：(G+L)→L
            nn.LayerNorm(self.hidden_size),  # 层归一化：稳定融合后的特征分布
            nn.ReLU(inplace=True)            # 非线性激活：增强特征表达能力
        )

        # 2. 轨迹位置预测层（loc）：预测未来每一步的2D坐标（x, y）
        # 输入：融合后的特征（hidden_size）
        # 输出：未来所有时间步的位置，维度为(future_steps * 2)（每步2个坐标）
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),  # 特征精炼：L→L
            nn.LayerNorm(self.hidden_size),  # 归一化：避免梯度爆炸
            nn.ReLU(inplace=True),            # 非线性激活
            nn.Linear(self.hidden_size, self.future_steps * 2)  # 输出位置：L→(H*2)
        )

        # 3. 轨迹不确定性预测层（scale）：仅当uncertain=True时启用
        # 预测未来每一步的位置不确定性（拉普拉斯分布的尺度参数），维度与位置对应
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),  # 特征精炼：L→L
                nn.LayerNorm(self.hidden_size),  # 归一化
                nn.ReLU(inplace=True),            # 非线性激活
                nn.Linear(self.hidden_size, self.future_steps * 2)  # 输出尺度：L→(H*2)
            )

        # 4. 模态概率预测层（pi）：预测每个智能体对应各轨迹模式的概率权重
        # 输入：局部特征+全局特征（与aggr_embed输入一致，确保概率基于完整特征）
        # 输出：每个模式的概率对数（后续通过softmax转为概率），维度为1（单值对应单个模式的权重）
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),  # (L+G)→L
            nn.LayerNorm(self.hidden_size),  # 归一化
            nn.ReLU(inplace=True),            # 激活
            nn.Linear(self.hidden_size, self.hidden_size),  # 再次精炼：L→L
            nn.LayerNorm(self.hidden_size),  # 归一化
            nn.ReLU(inplace=True),            # 激活
            nn.Linear(self.hidden_size, 1)   # 输出单模式权重：L→1
        )

        # 应用自定义权重初始化函数，确保所有线性层、MLP层参数初始状态合理
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 前向传播：输入局部特征和全局特征，输出多模态轨迹预测和模式概率
        # local_embed：局部编码器输出，形状[N, L]（N：智能体数，L：local_channels）
        # global_embed：全局交互模块输出，形状[F, N, G]（F：num_modes，G：global_channels）
        # 返回：(轨迹预测, 模式概率) → 轨迹预测含位置（+可选尺度），模式概率形状[N, F]

        # 1. 计算模态概率（pi）：为每个智能体的每个模式分配权重
        # 1.1 局部特征扩展：将local_embed从[N, L]扩展为[F, N, L]，与global_embed（[F, N, G]）维度对齐
        local_expand = local_embed.expand(self.num_modes, *local_embed.shape)
        # 1.2 特征拼接：将扩展后的局部特征与全局特征在最后一维拼接，形状[F, N, L+G]
        pi_input = torch.cat((local_expand, global_embed), dim=-1)
        # 1.3 预测概率：通过pi层输出每个模式的权重，形状[F, N, 1]→squeeze为[F, N]→转置为[N, F]（符合智能体-模式的维度顺序）
        pi = self.pi(pi_input).squeeze(-1).t()

        # 2. 计算轨迹预测（位置+可选尺度）
        # 2.1 特征融合：将扩展后的局部特征与全局特征拼接，输入aggr_embed得到融合特征，形状[F, N, L]
        out = self.aggr_embed(torch.cat((global_embed, local_expand), dim=-1))
        # 2.2 预测位置：通过loc层输出位置特征，形状[F, N, H*2]→reshape为[F, N, H, 2]（H：future_steps，2：x/y坐标）
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)

        # 2.3 （可选）预测不确定性：若uncertain=True，计算尺度参数并确保最小值
        if self.uncertain:
            # 预测尺度：通过scale层输出原始尺度→ELU激活（确保非负）→加1→加min_scale（避免过小）
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # 确保尺度≥min_scale，形状[F, N, H, 2]
            # 拼接位置和尺度：最后一维从2→4（x, y, scale_x, scale_y），返回轨迹预测和模式概率
            return torch.cat((loc, scale), dim=-1), pi  # 轨迹预测：[F, N, H, 4]；概率：[N, F]
        else:
            # 不预测不确定性：直接返回位置预测和模式概率
            return loc, pi  # 轨迹预测：[F, N, H, 2]；概率：[N, F]