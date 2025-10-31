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
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData


class ArgoverseV1Dataset(Dataset):
    # Argoverse V1数据集类：用于加载和预处理Argoverse自动驾驶轨迹预测数据集
    # 继承自PyTorch Geometric的Dataset类（或类似基础数据集类），核心功能是：
    # 1. 管理数据集的原始文件和预处理后文件路径；
    # 2. 提供数据集长度、单样本读取接口；
    # 3. 实现原始数据到模型输入格式的预处理（process方法）。

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        # 初始化数据集：配置数据集路径、划分、预处理参数
        # root：数据集根目录（如'/data/argoverse'）
        # split：数据集划分（'sample'/'train'/'val'/'test'，对应不同训练阶段）
        # transform：可选的数据增强函数（如随机旋转、噪声添加，默认None）
        # local_radius：局部区域半径（用于筛选智能体周围的邻居和车道，默认50米，与模型局部编码器匹配）
        
        # 1. 保存核心参数
        self._split = split  # 数据集划分（后续用于确定文件路径和预处理逻辑）
        self._local_radius = local_radius  # 局部区域半径（传递给预处理函数）
        # 2. 数据集原始压缩包的下载URL（Argoverse官方S3地址，用于自动下载）
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        
        # 3. 根据数据集划分确定子目录名称（与官方数据集解压后的目录结构一致）
        if split == 'sample':
            self._directory = 'forecasting_sample'  # 样本集目录（小批量数据，用于测试代码）
        elif split == 'train':
            self._directory = 'train'  # 训练集目录
        elif split == 'val':
            self._directory = 'val'    # 验证集目录
        elif split == 'test':
            self._directory = 'test_obs'  # 测试集目录（仅含观测数据，无真实轨迹）
        else:
            raise ValueError(split + ' is not valid')  # 无效划分时抛出异常
        
        # 4. 保存数据集根目录，后续拼接文件路径
        self.root = root
        
        # 5. 初始化原始文件和预处理后文件的路径列表
        self._raw_file_names = os.listdir(self.raw_dir)  # 获取原始数据目录下的所有文件名（如.json文件）
        # 预处理后文件：将原始文件名的后缀改为.pt（PyTorch二进制格式，加载更快）
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        # 预处理后文件的完整路径列表（拼接processed_dir和文件名）
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        
        # 6. 调用父类Dataset的初始化方法（传入根目录和数据增强函数）
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        # 只读属性：返回原始数据目录的完整路径
        # 结构：root/_directory/data（如'/data/argoverse/train/data'）
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        # 只读属性：返回预处理后数据目录的完整路径
        # 结构：root/_directory/processed（如'/data/argoverse/train/processed'）
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        # 父类Dataset要求的属性：返回原始文件名列表（用于自动检查原始文件是否存在）
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        # 父类Dataset要求的属性：返回预处理后文件名列表（用于自动检查预处理文件是否存在）
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        # 自定义属性：返回预处理后文件的完整路径列表（用于get方法加载数据）
        return self._processed_paths

    def process(self) -> None:
        # 核心方法：将原始数据（如.json）预处理为模型可直接输入的格式（保存为.pt文件）
        # 步骤：加载原始数据→提取智能体轨迹/车道信息→筛选局部区域→组织为TemporalData格式→保存
        
        # 1. 初始化Argoverse地图工具（用于获取车道信息，如车道坐标、转向限制）
        am = ArgoverseMap()
        
        # 2. 遍历所有原始文件（用tqdm显示预处理进度条）
        for raw_path in tqdm(self.raw_paths):
            # 2.1 调用预处理函数process_argoverse，处理单个原始文件
            # 输入：数据集划分、原始文件路径、地图工具、局部半径
            # 输出：包含"智能体轨迹、车道特征、序列ID"等的字典（kwargs）
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            
            # 2.2 将预处理结果组织为TemporalData格式（PyTorch Geometric的时序数据类，适配模型输入）
            data = TemporalData(**kwargs)
            
            # 2.3 保存预处理后的数据到processed_dir目录，文件名用序列ID（seq_id）命名（确保唯一性）
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        # 父类Dataset要求的方法：返回数据集的样本总数（即原始文件的数量）
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        # 父类Dataset要求的方法：根据索引idx读取单个预处理后的样本
        # 输入：样本索引idx（0~len-1）
        # 输出：预处理后的TemporalData对象（含智能体轨迹、车道特征等，可直接传入模型）
        return torch.load(self.processed_paths[idx])

'''
HiVT 数据预处理的核心流水线，完成了从 "原始 CSV" 到 "模型输入" 的全流程转换，关键价值如下：
1. 数据清洗与筛选：仅保留历史时间步可见的智能体，避免无效数据干扰模型；
2. 坐标系统一：以 AV 最后历史步为中心旋转坐标系，实现模型的旋转不变性（无论场景初始朝向如何，模型都能统一处理）；
3. 特征工程：将绝对坐标转换为相对位移（模型更关注运动趋势而非绝对位置），生成 BOS 掩码辅助时序编码；
4. 车道 - 智能体关联：通过get_lane_features筛选局部车道，提取车道属性（转向、交叉口）和相对向量，为 ALEncoder 提供输入；
5. 适配多场景：兼容训练 / 验证 / 测试集（测试集无 y 标签），支持不同城市地图的车道特征提取。

与模型的衔接
预处理输出的字典与TemporalData的参数完全对齐，后续只需通过TemporalData(**kwargs)即可创建模型输入实例，
'''
def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    # Argoverse原始数据预处理核心函数：将单条原始CSV数据（如场景轨迹）转换为TemporalData所需的结构化字典
    # 核心流程：数据筛选→坐标系变换→智能体特征处理→车道特征提取→标签组织，最终输出模型可直接使用的特征
    
    # 1. 加载原始数据并筛选有效智能体（仅保留历史时间步可见的智能体）
    df = pd.read_csv(raw_path)  # 加载原始CSV数据（每行含一个智能体在一个时间步的位置、类型等）
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))  # 提取场景内所有时间步并排序（共50步：20历史+30未来）
    historical_timestamps = timestamps[:20]  # 前20步为历史观测时间步（模型输入）
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]  # 筛选历史时间步的数据
    actor_ids = list(historical_df['TRACK_ID'].unique())  # 提取历史时间步可见的智能体ID（排除全程不可见的智能体）
    df = df[df['TRACK_ID'].isin(actor_ids)]  # 过滤原始数据：仅保留有效智能体
    num_nodes = len(actor_ids)  # 有效智能体总数（N）

    # 2. 定位关键智能体（AV自动驾驶车、AGENT目标智能体）和场景城市
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc  # 筛选AV（自动驾驶车辆）数据
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])  # AV在智能体列表中的索引
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc  # 筛选AGENT（主要预测目标，如其他车辆）数据
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])  # AGENT在智能体列表中的索引
    city = df['CITY_NAME'].values[0]  # 场景所在城市（如'MIA'迈阿密、'PIT'匹兹堡，用于加载对应地图）

    # 3. 坐标系变换：将场景中心设为AV的最后历史时间步位置，并旋转坐标系（实现旋转不变性）
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)  # 坐标原点：AV第19步（最后历史步）位置
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)  # AV运动方向向量
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])  # 旋转角度：使AV运动方向与X轴对齐
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],  # 2D旋转矩阵（逆时针旋转theta角）
                               [torch.sin(theta), torch.cos(theta)]])

    # 4. 初始化智能体核心特征张量
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)  # 智能体轨迹特征（N, 50步, 2维坐标）
    # 智能体间边索引：生成所有智能体对（i,j）i≠j，形状[2, N*(N-1)]（描述全连接的智能体交互图）
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)  # 填充掩码（True=无效填充，False=有效数据）
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)  # BOS令牌掩码（标记轨迹起始步，辅助时序编码）
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)  # 每个智能体的运动方向角度（用于局部旋转）

    # 5. 填充每个智能体的轨迹数据并计算运动方向
    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)  # 当前智能体在列表中的索引
        node_steps = [timestamps.index(t) for t in actor_df['TIMESTAMP']]  # 该智能体存在的时间步索引
        padding_mask[node_idx, node_steps] = False  # 标记有效时间步（非填充）
        
        # 若智能体在最后历史步（19步）不可见，则不预测其未来轨迹
        if padding_mask[node_idx, 19]:
            padding_mask[node_idx, 20:] = True  # 未来30步标记为填充
        
        # 转换坐标：原始坐标→以AV为中心的旋转后坐标
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        
        # 计算智能体运动方向（用最后两个历史有效步的差值近似）
        node_hist_steps = [s for s in node_steps if s < 20]  # 历史时间步内的有效步
        if len(node_hist_steps) > 1:
            heading_vec = x[node_idx, node_hist_steps[-1]] - x[node_idx, node_hist_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vec[1], heading_vec[0])
        else:
            padding_mask[node_idx, 20:] = True  # 有效历史步<2，不预测未来

    # 6. 生成BOS掩码（标记轨迹的"起始时刻"：前一步无效、当前步有效）
    bos_mask[:, 0] = ~padding_mask[:, 0]  # 第0步有效则为起始
    bos_mask[:, 1:20] = padding_mask[:, :19] & ~padding_mask[:, 1:20]  # 前一步无效且当前步有效

    # 7. 转换轨迹为"相对位移"特征（模型输入需相对位移，而非绝对坐标）
    positions = x.clone()  # 保存原始绝对坐标（用于后续计算）
    # 未来步（20-49）：相对位移 = 当前步坐标 - 最后历史步（19步）坐标（仅有效步计算）
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    # 历史步（1-19）：相对位移 = 当前步坐标 - 前一步坐标（仅有效步计算）
    x[:, 1:20] = torch.where((padding_mask[:, :19] | padding_mask[:, 1:20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1:20] - x[:, :19])
    x[:, 0] = torch.zeros(num_nodes, 2)  # 第0步无前置步，相对位移设为0

    # 8. 提取车道特征（调用get_lane_features，基于最后历史步的智能体位置筛选局部车道）
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]  # 最后历史步（19步）的智能体数据
    node_inds_19 = [actor_ids.index(aid) for aid in df_19['TRACK_ID']]  # 19步可见的智能体索引
    node_pos_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()  # 19步原始位置
    # 调用工具函数，获取局部半径内的车道特征（含车道向量、属性、车道-智能体连接）
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(am, node_inds_19, node_pos_19, origin, rotate_mat, city, radius)

    # 9. 组织标签（y）和序列ID（seq_id）
    y = None if split == 'test' else x[:, 20:]  # 测试集无真实轨迹（y=None），训练/验证集y为未来30步相对位移
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]  # 从文件名提取序列ID（如'12345'）

    # 10. 返回结构化特征字典（与TemporalData的__init__参数一一对应）
    return {
        'x': x[:, :20],  # 模型输入：历史20步相对位移（N,20,2）
        'positions': positions,  # 所有50步绝对坐标（N,50,2）
        'edge_index': edge_index,  # 智能体间边索引（2, N*(N-1)）
        'y': y,  # 标签：未来30步相对位移（N,30,2），测试集为None
        'num_nodes': num_nodes,  # 智能体总数N
        'padding_mask': padding_mask,  # 填充掩码（N,50）
        'bos_mask': bos_mask,  # BOS掩码（N,20）
        'rotate_angles': rotate_angles,  # 智能体运动方向角度（N）
        'lane_vectors': lane_vectors,  # 车道向量（L,2），L为车道段数
        'is_intersections': is_intersections,  # 车道是否为交叉口（L），0/1
        'turn_directions': turn_directions,  # 车道转向（L），0=直行/1=左转/2=右转
        'traffic_controls': traffic_controls,  # 车道是否有交通控制（L），0/1
        'lane_actor_index': lane_actor_index,  # 车道-智能体边索引（2, E_AL），E_AL为连接数
        'lane_actor_vectors': lane_actor_vectors,  # 车道-智能体相对向量（E_AL,2）
        'seq_id': int(seq_id),  # 序列ID（整数）
        'av_index': av_index,  # AV智能体索引
        'agent_index': agent_index,  # AGENT智能体索引
        'city': city,  # 场景城市
        'origin': origin.unsqueeze(0),  # 坐标原点（1,2）
        'theta': theta,  # 坐标系旋转角度
    }


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, ...]:
    # 车道特征提取工具函数：基于智能体位置筛选局部半径内的车道，并提取车道属性和车道-智能体交互
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()  # 存储局部区域内的唯一车道ID（避免重复）

    # 1. 筛选局部区域内的所有车道ID（基于每个智能体的位置，半径radius内）
    for node_pos in node_positions:
        # 调用ArgoverseMap工具，获取该智能体位置radius米内的所有车道ID
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_pos[0], node_pos[1], city, radius))
    
    # 2. 转换智能体位置到旋转后坐标系（与车道坐标对齐）
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()

    # 3. 提取每条车道的特征（向量、属性）
    for lane_id in lane_ids:
        # 获取车道中心线坐标（原始坐标系），取前2维（x,y）
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, :2]).float()
        # 转换车道中心线到旋转后坐标系
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        
        # 提取车道属性（调用ArgoverseMap工具）
        is_intersection = am.lane_is_in_intersection(lane_id, city)  # 是否在交叉口（布尔值）
        turn_direction = am.get_lane_turn_direction(lane_id, city)  # 转向方向（'NONE'/'LEFT'/'RIGHT'）
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)  # 是否有交通控制（布尔值）
        
        # 生成车道向量（中心线相邻点的差值，描述车道走向）
        lane_positions.append(lane_centerline[:-1])  # 车道段起点坐标（排除最后一个点）
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])  # 车道段向量（方向+长度）
        
        # 扩展车道属性到每个车道段（一条车道含多个段，每个段属性相同）
        count = len(lane_centerline) - 1  # 车道段数量（中心线点数量-1）
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))  # 0/1编码
        # 转向方向编码（0=直行，1=左转，2=右转）
        if turn_direction == 'NONE':
            turn_dir_code = 0
        elif turn_direction == 'LEFT':
            turn_dir_code = 1
        elif turn_direction == 'RIGHT':
            turn_dir_code = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_dir_code * torch.ones(count, dtype=torch.uint8))
        # 交通控制编码（0=无，1=有）
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))

    # 4. 拼接所有车道特征（从列表→张量）
    lane_positions = torch.cat(lane_positions, dim=0)  # 所有车道段起点坐标（L,2），L=总车道段数
    lane_vectors = torch.cat(lane_vectors, dim=0)  # 所有车道段向量（L,2）
    is_intersections = torch.cat(is_intersections, dim=0)  # 所有车道段交叉口属性（L）
    turn_directions = torch.cat(turn_directions, dim=0)  # 所有车道段转向属性（L）
    traffic_controls = torch.cat(traffic_controls, dim=0)  # 所有车道段交通控制属性（L）

    # 5. 生成车道-智能体的连接关系（边索引+相对向量）
    # 生成所有车道段与智能体的组合（全连接），形状[2, L*M]，M为智能体数
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    # 计算车道段起点到每个智能体的相对向量（车道坐标 - 智能体坐标）
    lane_actor_vectors = lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    
    # 筛选有效连接：仅保留相对距离<radius的车道-智能体对（排除远距离无效连接）
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius  # 距离掩码（True=有效）
    lane_actor_index = lane_actor_index[:, mask]  # 过滤后车道-智能体边索引（2, E_AL）
    lane_actor_vectors = lane_actor_vectors[mask]  # 过滤后相对向量（E_AL,2）

    # 返回所有车道特征
    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors