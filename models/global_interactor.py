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
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import TemporalData
from utils import init_weights


class GlobalInteractor(nn.Module):
    # 全局交互模块：捕捉跨局部区域的长距离依赖关系（如不同车道组的智能体交互）
    # 弥补局部编码器的感受野限制，让模型理解全局场景上下文（如交叉路口的远距离车辆影响）

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 edge_dim: int,
                 num_modes: int = 6,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 rotate: bool = True) -> None:
        # 初始化父类（nn.Module）
        super(GlobalInteractor, self).__init__()
        
        # 保存关键参数为实例变量
        self.historical_steps = historical_steps  # 历史时间步数（与局部编码器保持一致）
        self.embed_dim = embed_dim                # 特征嵌入维度（统一特征维度）
        self.num_modes = num_modes                # 多模态预测的轨迹模式数量（默认6种）

        # 1. 相对关系嵌入器：编码智能体之间的全局相对关系（跨局部区域的坐标系差异）
        if rotate:
            # 若启用旋转变换（rotate=True），需同时编码相对位置和角度差（均为edge_dim维度）
            # 用MultipleInputEmbedding融合两个连续特征，输出embed_dim维度的关系嵌入
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            # 若不启用旋转，仅编码相对位置（edge_dim维度），用SingleInputEmbedding处理
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        
        # 2. 全局交互层列表：堆叠多层GlobalInteractorLayer（论文中默认3层）
        # 每层通过注意力机制传递全局信息，逐步捕捉复杂的长距离依赖
        self.global_interactor_layers = nn.ModuleList(
            [GlobalInteractorLayer(
                embed_dim=embed_dim,  # 嵌入维度
                num_heads=num_heads,  # 注意力头数
                dropout=dropout       # Dropout概率
            ) for _ in range(num_layers)]  # 堆叠num_layers层
        )
        
        # 3. 层归一化：对多层全局交互后的特征做归一化，稳定输出分布
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. 多模式投影层：将全局特征映射到多模态空间（为每个模式生成独立特征）
        # 输入：[N, embed_dim]（N为智能体数），输出：[N, num_modes * embed_dim]
        # 后续会拆分为[num_modes, N, embed_dim]，对应每种模式的特征
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        
        # 应用自定义权重初始化函数，确保各层参数初始状态合理
        self.apply(init_weights)

    def forward(self,
                data: TemporalData,
                local_embed: torch.Tensor) -> torch.Tensor:
        edge_index, _ = subgraph(subset=~data['padding_mask'][:, self.historical_steps - 1], edge_index=data.edge_index)
        rel_pos = data['positions'][edge_index[0], self.historical_steps - 1] - data['positions'][
            edge_index[1], self.historical_steps - 1]
        if data['rotate_mat'] is None:
            rel_embed = self.rel_embed(rel_pos)
        else:
            rel_pos = torch.bmm(rel_pos.unsqueeze(-2), data['rotate_mat'][edge_index[1]]).squeeze(-2)
            rel_theta = data['rotate_angles'][edge_index[0]] - data['rotate_angles'][edge_index[1]]
            rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
            rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
            rel_embed = self.rel_embed([rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)])
        x = local_embed
        for layer in self.global_interactor_layers:
            x = layer(x, edge_index, rel_embed)
        x = self.norm(x)  # [N, D]
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # [N, F, D]
        x = x.transpose(0, 1)  # [F, N, D]
        return x


class GlobalInteractorLayer(MessagePassing):
    # 全局交互层：GlobalInteractor的核心基础层，单一层负责一次全局注意力交互
    # 继承自MessagePassing，通过"消息传递"机制实现跨智能体的全局信息交换
    # 核心是融合智能体特征（节点特征）和智能体间关系特征（边特征），捕捉长距离依赖

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,** kwargs) -> None:
        # 初始化父类MessagePassing：聚合方式"add"（求和），节点维度0（按智能体维度聚合）
        super(GlobalInteractorLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        
        # 保存关键参数：嵌入维度、多头注意力头数（控制并行捕捉不同交互模式）
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. 注意力Query/Key/Value投影层：分"节点特征"和"边特征"单独投影
        # Query仅来自目标智能体（x_i）的节点特征（无需边特征）
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)  # 目标智能体特征→Query
        # Key分两部分：源智能体（x_j）的节点特征 + 两智能体间的边特征（相对关系）
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)  # 源智能体特征→Key节点部分
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)  # 边特征→Key边部分
        # Value同理：源智能体节点特征 + 边特征，确保注意力值融合双方关系
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)  # 源智能体特征→Value节点部分
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)  # 边特征→Value边部分
        
        # 2. 自连接投影层：保留目标智能体自身原始特征（残差连接的关键）
        # 避免全局交互时丢失智能体自身的局部运动趋势
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        
        # 3. 注意力dropout：对注意力权重随机失活，防止过拟合到特定全局交互模式
        self.attn_drop = nn.Dropout(dropout)
        
        # 4. 门控更新线性层：计算门控系数（控制全局信息的融合比例）
        # 比如对无全局交互的智能体（如孤立车辆），门控系数趋近0，减少无效全局信息干扰
        self.lin_ih = nn.Linear(embed_dim, embed_dim)  # 输入：全局聚合的消息（inputs）
        self.lin_hh = nn.Linear(embed_dim, embed_dim)  # 输入：目标智能体自身特征（x）
        
        # 5. 注意力输出投影层：将多头注意力的拼接结果映射回embed_dim维度
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 6. 输出dropout：对投影后的全局特征失活，进一步防止过拟合
        self.proj_drop = nn.Dropout(dropout)
        
        # 7. 层归一化：稳定训练（分别用于注意力层前、MLP层前，避免梯度爆炸）
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 8. 前馈网络（MLP）：对注意力融合后的全局特征做非线性变换
        # 扩大特征感受野，捕捉复杂全局交互模式（如"多智能体协同变道"）
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 升维：增强特征表达能力
            nn.ReLU(inplace=True),                # 非线性激活：引入复杂交互
            nn.Dropout(dropout),                  # dropout失活：防止过拟合
            nn.Linear(embed_dim * 4, embed_dim),  # 降维：映射回原嵌入维度
            nn.Dropout(dropout)                   # 再次失活：强化泛化能力
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor,
                size: Size = None) -> torch.Tensor:
        # 前向传播主逻辑："注意力交互→残差连接→MLP→残差连接"的标准Transformer层结构
        # x：输入节点特征（形状[N, embed_dim]，N为智能体数）
        # edge_index：边索引（描述智能体间的连接关系，如[2, E]，E为边数）
        # edge_attr：边特征（智能体间的相对关系，形状[E, embed_dim]）
        # size：可选，指定节点数量（用于非对称图）
        
        # 第一步：注意力交互（带残差连接）
        # 先对x做层归一化（norm1），再通过_mha_block计算全局注意力消息，最后与原始x残差相加
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr, size)
        
        # 第二步：MLP非线性变换（带残差连接）
        # 对第一步结果做层归一化（norm2），再通过_ff_block做MLP变换，最后与当前x残差相加
        x = x + self._ff_block(self.norm2(x))
        
        # 返回更新后的全局特征（融合了长距离依赖）
        return x

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        # MessagePassing核心方法：计算从源智能体（j）到目标智能体（i）的"消息"
        # x_i：目标智能体特征（[E, embed_dim]，每个边对应一个目标智能体）
        # x_j：源智能体特征（[E, embed_dim]，每个边对应一个源智能体）
        # edge_attr：边特征（[E, embed_dim]，每个边的相对关系）
        # index/ptr/size_i：MessagePassing内部用于聚合的辅助参数（如目标智能体索引）
        
        # 1. 多头注意力拆分：将特征按头数拆分（形状从[E, embed_dim]→[E, num_heads, embed_dim//num_heads]）
        # Query：仅来自目标智能体x_i（关注"我需要什么信息"）
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # Key：融合源智能体x_j和边特征（关注"源智能体有什么信息+我们的关系"）
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # Value：同样融合源智能体x_j和边特征（关注"源智能体提供什么价值信息+关系权重"）
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        
        # 2. 计算注意力权重（缩放点积注意力）
        scale = (self.embed_dim // self.num_heads) ** 0.5  # 缩放因子：避免梯度消失
        # Key融合（节点+边），与Query做点积后除以缩放因子
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        # 按目标智能体分组做softmax（确保每个目标智能体的注意力权重和为1）
        alpha = softmax(alpha, index, ptr, size_i)
        # 注意力dropout：随机屏蔽部分权重，防止过拟合
        alpha = self.attn_drop(alpha)
        
        # 3. 计算消息：Value融合（节点+边）乘以注意力权重，得到源智能体传递给目标的信息
        return (value_node + value_edge) * alpha.unsqueeze(-1)  # unsqueeze(-1)：权重与Value维度对齐

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        # MessagePassing核心方法：将聚合后的消息（inputs）与目标智能体自身特征（x）融合
        # inputs：聚合后的全局消息（[N, num_heads, embed_dim//num_heads]→经reshape后[N, embed_dim]）
        # x：目标智能体原始特征（[N, embed_dim]）
        
        # 1. 消息形状恢复：将多头注意力拆分的特征重组为[N, embed_dim]
        inputs = inputs.view(-1, self.embed_dim)
        
        # 2. 门控更新：计算门控系数（0~1），控制自身特征与全局消息的融合比例
        # 门控系数=σ(全局消息投影 + 自身特征投影)，σ为sigmoid函数
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        
        # 3. 特征融合：全局消息 + 门控*(自身特征投影 - 全局消息)
        # 本质是残差连接的变体：既保留全局消息，又通过门控调节自身特征的贡献
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(self,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   size: Size) -> torch.Tensor:
        # 封装多头注意力流程：调用MessagePassing的propagate方法触发message→aggregate→update
        # propagate：自动完成"消息计算→按目标智能体聚合→特征更新"的全流程
        # 输出：更新后的全局特征→经out_proj投影→proj_drop失活
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # 封装MLP流程：直接调用mlp对输入特征做非线性变换
        return self.mlp(x)
