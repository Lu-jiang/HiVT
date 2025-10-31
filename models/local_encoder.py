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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import DistanceDropEdge
from utils import TemporalData
from utils import init_weights


class LocalEncoder(nn.Module):
    # 局部编码器：以单个智能体为中心，提取局部区域内的特征（智能体交互、时间依赖、车道交互）
    # 继承自PyTorch的nn.Module，是模型的核心组件之一

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False) -> None:
        # 初始化父类（nn.Module）
        super(LocalEncoder, self).__init__()
        
        # 保存关键参数为实例变量
        self.historical_steps = historical_steps  # 历史时间步数（用于时间维度处理）
        self.parallel = parallel                  # 是否并行计算多个智能体的局部特征
        
        # 初始化距离筛选器：过滤超出局部区域的智能体/车道
        # 仅保留中心智能体周围local_radius（默认50米）内的邻居智能体和车道段
        self.drop_edge = DistanceDropEdge(local_radius)
        
        # 初始化智能体-智能体交互编码器（AAEncoder）
        # 负责建模局部区域内中心智能体与邻居智能体的交互关系
        self.aa_encoder = AAEncoder(
            historical_steps=historical_steps,  # 历史时间步数
            node_dim=node_dim,                  # 智能体轨迹向量维度
            edge_dim=edge_dim,                  # 智能体间相对位置向量维度
            embed_dim=embed_dim,                # 嵌入维度
            num_heads=num_heads,                # 注意力头数
            dropout=dropout,                    # Dropout概率
            parallel=parallel                   # 是否并行计算
        )
        
        # 初始化时间编码器（TemporalEncoder）
        # 负责捕捉智能体历史轨迹的时间依赖关系（如速度变化、加速度趋势）
        self.temporal_encoder = TemporalEncoder(
            historical_steps=historical_steps,  # 历史时间步数
            embed_dim=embed_dim,                # 嵌入维度
            num_heads=num_heads,                # 注意力头数
            dropout=dropout,                    # Dropout概率
            num_layers=num_temporal_layers      # Transformer层数（默认4层）
        )
        
        # 初始化智能体-车道交互编码器（ALEncoder）
        # 负责建模局部区域内智能体与车道的交互关系（如车道约束、转向引导）
        self.al_encoder = ALEncoder(
            node_dim=node_dim,                  # 智能体轨迹向量维度（此处复用，实际处理车道特征）
            edge_dim=edge_dim,                  # 智能体-车道相对位置向量维度
            embed_dim=embed_dim,                # 嵌入维度
            num_heads=num_heads,                # 注意力头数
            dropout=dropout                     # Dropout概率
        )

    def forward(self, data: TemporalData) -> torch.Tensor:
        # 局部编码器前向传播：提取智能体的时序特征、智能体间局部交互、智能体-车道交互
        # 输入：TemporalData实例（含历史轨迹、边索引、车道特征等）
        # 输出：local_embed（融合局部信息的智能体特征），形状[N, D]（N=智能体数，D=特征维度）

        # 1. 预处理每个历史时间步的智能体间边信息（动态构建每步的交互图）
        for t in range(self.historical_steps):  # 遍历所有历史时间步（如20步）
            # 1.1 筛选有效边：基于当前步的padding_mask，保留"两端智能体均有效"的边
            # subset=~data['padding_mask'][:, t]：当前步t有效的智能体掩码（True=有效）
            # 返回筛选后的边索引edge_index_t和掩码（此处用_忽略掩码）
            data[f'edge_index_{t}'], _ = subgraph(
                subset=~data['padding_mask'][:, t],  # 有效智能体掩码
                edge_index=data.edge_index  # 原始全连接边索引
            )
            
            # 1.2 计算当前步的边属性（智能体间相对位置向量）
            # 边起点位置 - 边终点位置 → 描述智能体i到j的相对位置
            data[f'edge_attr_{t}'] = (
                data['positions'][data[f'edge_index_{t}'][0], t]  # 边起点（i）在t步的位置
                - data['positions'][data[f'edge_index_{t}'][1], t]  # 边终点（j）在t步的位置
            )

        # 2. 智能体-智能体交互编码（AAEncoder）：建模同一步内智能体间的局部交互
        if self.parallel:  # 并行模式：所有时间步同时输入编码器（效率更高）
            # 2.1 构建每个时间步的子图数据（Data对象）
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                # 随机丢弃部分边（数据增强，降低过拟合风险）
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                # 构建t步的子图：含当前步特征x、边索引、边属性
                snapshots[t] = Data(
                    x=data.x[:, t],  # t步的智能体特征（相对位移），形状[N, 2]
                    edge_index=edge_index,  # t步的有效边索引，形状[2, E_t]
                    edge_attr=edge_attr,  # t步的边属性（相对向量），形状[E_t, 2]
                    num_nodes=data.num_nodes  # 智能体总数N
                )
            
            # 2.2 批量拼接所有时间步的子图（Batch对象）
            batch = Batch.from_data_list(snapshots)  # 自动处理不同时间步的边索引偏移
            
            # 2.3 并行编码所有时间步的智能体-智能体交互
            out = self.aa_encoder(
                x=batch.x,  # 批量特征，形状[T*N, 2]（T=时间步，N=智能体数）
                t=None,  # 并行模式不单独传入时间步（内部自动处理）
                edge_index=batch.edge_index,  # 批量边索引，形状[2, sum(E_t)]
                edge_attr=batch.edge_attr,  # 批量边属性，形状[sum(E_t), 2]
                bos_mask=data['bos_mask'],  # BOS掩码，形状[N, T]（标记轨迹起始）
                rotate_mat=data['rotate_mat']  # 旋转矩阵，形状[N, 2, 2]（若启用旋转增强）
            )
            
            # 2.4 调整输出形状：[T*N, D] → [T, N, D]（恢复时间步和智能体维度）
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        
        else:  # 串行模式：逐时间步输入编码器（内存占用更低）
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                # 随机丢弃部分边
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                
                # 逐步编码智能体-智能体交互
                out[t] = self.aa_encoder(
                    x=data.x[:, t],  # t步特征，形状[N, 2]
                    t=t,  # 当前时间步索引（用于时序编码）
                    edge_index=edge_index,  # t步边索引，形状[2, E_t]
                    edge_attr=edge_attr,  # t步边属性，形状[E_t, 2]
                    bos_mask=data['bos_mask'][:, t],  # t步的BOS掩码，形状[N]
                    rotate_mat=data['rotate_mat']  # 旋转矩阵
                )  # 输出：[N, D]
            
            # 拼接所有时间步的输出：[T, N, D]
            out = torch.stack(out)

        # 3. 时序编码（Temporal Encoder）：建模智能体自身的时序依赖（历史轨迹趋势）
        out = self.temporal_encoder(
            x=out,  # 输入：[T, N, D]（AAEncoder的输出）
            padding_mask=data['padding_mask'][:, :self.historical_steps]  # 时序掩码，形状[N, T]（标记无效时间步）
        )  # 输出：[N, D']（融合时序信息后的智能体特征）

        # 4. 智能体-车道交互编码（ALEncoder）：融合车道特征与智能体特征
        # 4.1 随机丢弃部分车道-智能体边（数据增强）
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        
        # 4.2 编码智能体与车道的交互（如车道约束、转向引导）
        out = self.al_encoder(
            x=(data['lane_vectors'], out),  # 输入：(车道特征[L, 2], 智能体特征[N, D'])
            edge_index=edge_index,  # 车道-智能体边索引，形状[2, E_AL]
            edge_attr=edge_attr,  # 边属性（相对向量），形状[E_AL, 2]
            is_intersections=data['is_intersections'],  # 车道交叉口属性[L]
            turn_directions=data['turn_directions'],  # 车道转向属性[L]
            traffic_controls=data['traffic_controls'],  # 车道交通控制属性[L]
            rotate_mat=data['rotate_mat']  # 旋转矩阵（若启用）
        )  # 输出：[N, D'']（融合车道信息的最终局部特征）

        return out  # 局部嵌入特征，供后续全局交互模块使用


class AAEncoder(MessagePassing):
    # 智能体-智能体交互编码器（Agent-Agent Encoder）
    # 继承自PyTorch Geometric的MessagePassing，用于建模局部区域内中心智能体与邻居智能体的交互
    # 核心是通过旋转不变的交叉注意力机制融合邻居信息

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,** kwargs) -> None:
        # 初始化父类MessagePassing：聚合方式为"add"（求和），节点维度为0（按智能体维度聚合）
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        
        # 保存关键参数为实例变量
        self.historical_steps = historical_steps  # 历史时间步数（用于时间维度特征处理）
        self.embed_dim = embed_dim                # 嵌入维度
        self.num_heads = num_heads                # 多头注意力头数
        self.parallel = parallel                  # 是否并行处理多个中心智能体的邻居交互
        
        # 中心智能体轨迹嵌入器：将中心智能体的原始轨迹向量（node_dim维度）编码为嵌入特征（embed_dim维度）
        # SingleInputEmbedding是带LayerNorm和激活函数的MLP
        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        
        # 邻居智能体嵌入器：同时处理邻居的轨迹向量（node_dim）和与中心的相对位置（edge_dim）
        # 输入为两个特征（轨迹+相对位置），输出融合后的嵌入特征（embed_dim维度）
        self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        
        # 注意力Query/Key/Value投影层：将嵌入特征映射到注意力空间
        self.lin_q = nn.Linear(embed_dim, embed_dim)  # Query投影（中心智能体特征）
        self.lin_k = nn.Linear(embed_dim, embed_dim)  # Key投影（邻居智能体特征）
        self.lin_v = nn.Linear(embed_dim, embed_dim)  # Value投影（邻居智能体特征）
        
        # 自连接投影层：保留中心智能体自身特征（残差连接的一部分）
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        
        # 注意力 dropout：防止注意力权重过拟合
        self.attn_drop = nn.Dropout(dropout)
        
        # 门控更新的线性层：用于计算门控系数（控制邻居信息的融合比例）
        # 类似GRU的更新门机制，决定保留多少自身特征、融合多少邻居特征
        self.lin_ih = nn.Linear(embed_dim, embed_dim)  # 输入特征（邻居聚合信息）投影
        self.lin_hh = nn.Linear(embed_dim, embed_dim)  # 隐藏状态（自身特征）投影
        
        # 输出投影层：将多头注意力的结果映射回嵌入维度
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出 dropout：防止输出特征过拟合
        self.proj_drop = nn.Dropout(dropout)
        
        # 层归一化：稳定训练，加速收敛（分别用于注意力后和MLP后）
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 多层感知机（MLP）：用于特征非线性变换（注意力后的前馈网络）
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 升维（扩大感受野）
            nn.ReLU(inplace=True),                # 非线性激活
            nn.Dropout(dropout),                  # dropout防止过拟合
            nn.Linear(embed_dim * 4, embed_dim),  # 降维回原嵌入维度
            nn.Dropout(dropout)                   # 再次dropout
        )
        
        # BOS（Begin of Sequence）令牌：用于标记轨迹的起始，辅助时间维度特征对齐
        # 形状为[historical_steps, embed_dim]，与每个时间步的特征维度匹配
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)  # 初始化BOS令牌参数
        
        # 应用自定义权重初始化函数（如 Xavier 初始化），确保网络稳定启动
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        if self.parallel:
            if rotate_mat is None:
                center_embed = self.center_embed(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1))
            else:
                center_embed = self.center_embed(
                    torch.matmul(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1).unsqueeze(-2),
                                 rotate_mat.expand(self.historical_steps, *rotate_mat.shape)).squeeze(-2))
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                       self.bos_token.unsqueeze(-2),
                                       center_embed).reshape(x.shape[0], -1)
                                    #    center_embed).view(x.shape[0], -1)
        else:
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2))
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat,
                                                      size)
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):
    # 时间编码器：捕捉智能体历史轨迹的时间依赖关系（如速度变化、运动趋势）
    # 基于Transformer Encoder实现，核心是通过带时间掩码的自注意力建模时序顺序

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        # 初始化父类（nn.Module）
        super(TemporalEncoder, self).__init__()
        
        # 1. 初始化单个Transformer编码器层（TemporalEncoderLayer）
        # 该层包含时间自注意力和前馈网络，是构建Transformer Encoder的基础单元
        # embed_dim：特征嵌入维度，num_heads：注意力头数，dropout：防止过拟合的dropout概率
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 2. 构建多层Transformer Encoder
        # 堆叠num_layers个TemporalEncoderLayer（论文中默认4层），最后加LayerNorm做输出归一化
        # 作用：逐步捕捉不同尺度的时间依赖（如短期速度变化、长期运动趋势）
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,  # 基础编码器层
            num_layers=num_layers,        # 堆叠层数（默认4层）
            norm=nn.LayerNorm(embed_dim)  # 输出归一化，稳定多层堆叠后的特征分布
        )
        
        # 3. 初始化特殊令牌（Token）：用于时序特征对齐和全局信息聚合
        # 3.1 填充令牌（padding_token）：处理轨迹中可能存在的缺失时间步（如智能体后期才进入场景）
        # 形状：[historical_steps, 1, embed_dim] → 每个时间步对应一个填充向量，适配批次维度
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        # 3.2 CLS令牌（cls_token）：聚合整个历史轨迹的时序信息，输出全局时间特征
        # 形状：[1, 1, embed_dim] → 单个向量，用于汇总所有时间步的特征
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        # 3.3 位置嵌入（pos_embed）：为每个时间步添加位置信息，让Transformer感知时序顺序
        # 形状：[historical_steps + 1, 1, embed_dim] → 额外+1是为了容纳CLS令牌（共historical_steps+1个token）
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        
        # 4. 生成时间掩码（attn_mask）：防止未来时间步的信息泄露
        # 调用自定义函数生成下三角掩码（下三角为0，上三角为-∞），确保当前时间步只关注之前的时间步
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        # 将掩码注册为buffer（不参与梯度更新的固定参数），避免每次前向传播重复生成
        self.register_buffer('attn_mask', attn_mask)
        
        # 5. 初始化特殊令牌的参数：用正态分布初始化（均值0，标准差0.02），符合Transformer参数初始化惯例
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        
        # 6. 应用自定义权重初始化函数（如Xavier初始化），确保所有层参数初始状态合理
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        # 生成时间序列的后续掩码（subsequent mask），用于Transformer的时间注意力机制
        # 核心作用：防止模型在处理第t个时间步时，关注到t之后的未来时间步信息（避免信息泄露）

        # 步骤1：生成上三角矩阵（对角线及以上为1，其余为0）
        # torch.ones(seq_len, seq_len) → 创建一个[seq_len×seq_len]的全1矩阵
        # torch.triu(..., diagonal=0) → 取上三角部分（包含对角线），下三角置0
        # == 1 → 转换为布尔矩阵（上三角为True，下三角为False）
        # transpose(0, 1) → 矩阵转置（确保掩码维度与注意力计算的查询-键维度对齐）
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        
        # 步骤2：将布尔矩阵转换为注意力掩码（未来时间步设为-∞，当前及过去时间步设为0）
        # masked_fill(mask == 0, float('-inf')) → 下三角（False）对应未来时间步，掩码设为-∞
        # masked_fill(mask == 1, float(0.0)) → 上三角（True）对应当前及过去时间步，掩码设为0
        # 注：在注意力计算中，softmax(-∞)≈0，即未来时间步的注意力权重被屏蔽
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # 返回生成的掩码矩阵，形状为[seq_len, seq_len]
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal = None) -> torch.Tensor:    # torch 新版本带来的 因果注意力 入参
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class ALEncoder(MessagePassing):
    # 智能体-车道交互编码器（Agent-Lane Encoder）
    # 继承自PyTorch Geometric的MessagePassing，核心是建模局部区域内智能体与车道的交互关系
    # 比如车道对智能体运动的约束（如沿车道行驶、转向限制），为轨迹预测提供地图先验

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,** kwargs) -> None:
        # 初始化父类MessagePassing：聚合方式为"add"（求和），节点维度为0（按智能体维度聚合）
        # 与AAEncoder一致，确保多智能体特征聚合的一致性
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        
        # 保存关键参数为实例变量，用于后续层初始化和前向传播
        self.embed_dim = embed_dim                # 特征嵌入维度（统一输出维度）
        self.num_heads = num_heads                # 多头注意力头数（并行捕捉不同车道交互模式）

        # 1. 车道特征嵌入器：处理车道的连续型特征（如车道向量、智能体-车道相对位置）
        # 输入为两个连续特征：车道原始向量（node_dim维度，复用AAEncoder的node_dim命名）、
        # 智能体与车道的相对位置（edge_dim维度），输出统一的embed_dim维度嵌入
        # 复用之前的MultipleInputEmbedding，确保多输入特征编码逻辑一致
        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)

        # 2. 注意力机制核心层：将智能体特征与车道特征映射到注意力空间
        self.lin_q = nn.Linear(embed_dim, embed_dim)  # Query投影（输入：智能体的时空特征，输出：注意力查询向量）
        self.lin_k = nn.Linear(embed_dim, embed_dim)  # Key投影（输入：编码后的车道特征，输出：注意力键向量）
        self.lin_v = nn.Linear(embed_dim, embed_dim)  # Value投影（输入：编码后的车道特征，输出：注意力值向量）
        
        # 3. 自连接投影层：保留智能体自身特征（残差连接的关键）
        # 避免注意力融合车道信息时，丢失智能体原本的运动趋势
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        
        # 4. 注意力dropout：对注意力权重随机失活，防止过拟合到特定车道交互模式
        self.attn_drop = nn.Dropout(dropout)
        
        # 5. 门控更新线性层：计算门控系数（类似AAEncoder的门控机制）
        # 控制车道信息的融合比例（比如在无车道区域，门控系数趋近0，减少车道特征干扰）
        self.lin_ih = nn.Linear(embed_dim, embed_dim)  # 输入：车道聚合特征的投影
        self.lin_hh = nn.Linear(embed_dim, embed_dim)  # 输入：智能体自身特征的投影
        
        # 6. 注意力输出投影层：将多头注意力的拼接结果映射回embed_dim维度
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 7. 输出dropout：对投影后的特征失活，进一步防止过拟合
        self.proj_drop = nn.Dropout(dropout)
        
        # 8. 层归一化：稳定训练过程（分别用于注意力层后、MLP层后，避免梯度爆炸）
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 9. 前馈网络（MLP）：对注意力融合后的特征做非线性变换
        # 扩大特征感受野，捕捉智能体-车道交互的复杂模式（如"车道转向+智能体速度"的组合影响）
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 升维：将特征维度扩大4倍，增强表达能力
            nn.ReLU(inplace=True),                # 非线性激活：引入非线性交互
            nn.Dropout(dropout),                  # dropout失活：防止过拟合
            nn.Linear(embed_dim * 4, embed_dim),  # 降维：将特征映射回原embed_dim维度
            nn.Dropout(dropout)                   # 再次失活：强化泛化能力
        )

        # 10. 车道离散属性嵌入：将车道的离散语义信息编码为向量（与连续特征融合）
        # 10.1 车道是否为交叉口（is_intersection）：2类（是/否），对应2个嵌入向量
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        # 10.2 车道转向方向（turn_direction）：3类（直行/左转/右转，或其他分类），对应3个嵌入向量
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        # 10.3 车道交通控制（traffic_control）：2类（有信号灯/无信号灯，或其他分类），对应2个嵌入向量
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))

        # 11. 初始化离散属性嵌入参数：用正态分布（均值0，标准差0.02）初始化
        # 符合Transformer类模型的参数初始化惯例，确保初始特征分布稳定
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)

        # 12. 应用自定义权重初始化函数（如Xavier初始化）：确保所有线性层、MLP层参数初始合理
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor,
                turn_directions: torch.Tensor,
                traffic_controls: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
                                            turn_directions, traffic_controls, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections_j,
                turn_directions_j,
                traffic_controls_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                   torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   is_intersections: torch.Tensor,
                   turn_directions: torch.Tensor,
                   traffic_controls: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                               is_intersections=is_intersections, turn_directions=turn_directions,
                                               traffic_controls=traffic_controls, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)
