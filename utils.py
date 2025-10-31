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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

'''
数据流向示例
一个 TemporalData 实例对应 “一个场景的完整数据”，模型训练时的输入链路为：
    ArgoverseV1Dataset.get(idx) → 返回TemporalData实例 → DataLoader批处理 
    → 批次TemporalData传入HiVT.forward() → 模型提取各属性完成计算
'''
class TemporalData(Data):
    # 时序图数据类：继承自PyTorch Geometric的基础Data类
    # 核心作用是“统一封装时序轨迹+图结构数据”，适配HiVT模型“时序Transformer+图注意力”的混合输入需求
    # 既包含智能体的时序运动特征，也包含智能体/车道的图连接关系，同时兼容多模态预测所需的辅助信息

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,** kwargs) -> None:
        # 空数据处理：若未传入核心特征x，创建空实例（用于占位或初始化，无实际数据）
        if x is None:
            super(TemporalData, self).__init__()
            return
        
        # 1. 调用父类Data初始化，注册核心数据特征
        # 所有参数会成为实例属性，可通过“data.属性名”直接访问（如data.x、data.lane_vectors）
        super(TemporalData, self).__init__(
            x=x,  # 智能体时序特征：形状[N, T, 2]（N=智能体数，T=历史时间步，2=相对位移x/y）
            positions=positions,  # 智能体绝对坐标：形状[N, 50, 2]（50=20历史+30未来步，存原始位置用于后续计算）
            edge_index=edge_index,  # 智能体间图边索引：形状[2, E]（E=边数，描述智能体i→j的交互连接）
            y=y,  # 未来轨迹标签：形状[N, F, 2]（F=未来时间步，仅训练/验证集有，测试集为None）
            num_nodes=num_nodes,  # 智能体总数N（避免重复计算x.size(0)，提升效率）
            padding_mask=padding_mask,  # 填充掩码：形状[N, 50]（True=该步为无效填充，False=有效数据）
            bos_mask=bos_mask,  # BOS令牌掩码：形状[N, T]（标记轨迹起始步，辅助时序编码器定位）
            rotate_angles=rotate_angles,  # 智能体旋转角度：形状[N]（每个智能体的局部坐标系角度，实现旋转不变性）
            lane_vectors=lane_vectors,  # 车道向量特征：形状[L, 2]（L=车道段数，2=车道走向x/y向量）
            is_intersections=is_intersections,  # 车道交叉口属性：形状[L]（0=非交叉口，1=交叉口，用于ALEncoder）
            turn_directions=turn_directions,  # 车道转向属性：形状[L]（0=直行，1=左转，2=右转，用于ALEncoder）
            traffic_controls=traffic_controls,  # 车道交通控制属性：形状[L]（0=无，1=有信号灯，用于ALEncoder）
            lane_actor_index=lane_actor_index,  # 车道-智能体边索引：形状[2, E_AL]（E_AL=连接数，描述车道→智能体的交互）
            lane_actor_vectors=lane_actor_vectors,  # 车道-智能体相对向量：形状[E_AL, 2]（智能体与车道的位置关系）
            seq_id=seq_id,  # 数据序列ID（唯一标识一个场景样本，用于日志记录和结果追溯）
            **kwargs  # 自定义扩展特征（如场景城市、坐标原点等，灵活适配额外需求）
        )
        
        # 2. 处理时序边属性：edge_attrs是“每个时间步的智能体间边特征”列表
        # 因智能体间的相对关系（如距离、角度）随时间变化，需按时间步单独存储边特征
        if edge_attrs is not None:
            for t in range(self.x.size(1)):  # 遍历所有历史时间步T
                # 注册为实例属性：data.edge_attr_0（第0步）、data.edge_attr_1（第1步）...
                self[f'edge_attr_{t}'] = edge_attrs[t]  # 每个元素形状[E, 2]（E=边数，2=相对关系特征）

    def __inc__(self, key, value):
        # 重载父类Data的__inc__方法：定义“边索引递增规则”，适配PyTorch Geometric的批处理（batching）逻辑
        # 核心作用：当多个样本拼接成批次时，确保边索引指向正确的节点（避免不同样本的节点ID冲突）
        
        # 特殊处理“车道-智能体边索引”（lane_actor_index）：
        # 该边索引的第0维是“车道段ID”（来自lane_vectors），第1维是“智能体ID”（来自num_nodes）
        # 拼接时需分别按“车道段数”和“智能体数”递增，避免ID重叠
        if key == 'lane_actor_index':
            # 返回递增基数：[车道段总数, 智能体总数]
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            # 其他边索引（如智能体间edge_index）：使用父类默认规则（按智能体数num_nodes递增）
            return super().__inc__(key, value)


class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
