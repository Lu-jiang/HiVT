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
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class ADE(Metric):
    # ADE（Average Displacement Error，平均位移误差）评价指标
    # 用于量化多智能体轨迹预测的精度：计算预测轨迹与真实轨迹在所有未来时间步的平均L2距离
    # 继承自PyTorch Lightning的Metric类，支持分布式训练中的指标同步（如多GPU训练时聚合各卡结果）

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        # 初始化父类Metric：配置指标计算时机和分布式同步参数
        super(ADE, self).__init__(
            compute_on_step=compute_on_step,  # 是否在每一步（step）计算指标（默认True，实时更新）
            dist_sync_on_step=dist_sync_on_step,  # 是否在每一步同步分布式训练的指标（默认False，节省通信开销）
            process_group=process_group,  # 分布式训练的进程组（可选，默认使用全局进程组）
            dist_sync_fn=dist_sync_fn  # 自定义分布式同步函数（可选，默认使用框架自带同步逻辑）
        )
        
        # 注册状态变量：用于累积计算指标的中间结果（支持分布式同步）
        # 1. sum：累积所有样本的ADE总和（初始为0.0，分布式训练时按"求和"方式同步）
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        # 2. count：累积样本数量（初始为0，分布式训练时按"求和"方式同步，统计总智能体数）
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        # 更新指标状态：每批数据（或每个step）计算当前批次的ADE贡献，累加到sum和count
        # pred：模型预测的轨迹，形状通常为[F, N, H, 2]或[N, H, 2]（F=模态数，N=智能体数，H=未来步，2=x/y坐标）
        # target：真实轨迹，形状通常为[N, H, 2]（与pred的位置维度对应）
        
        # 分步计算当前批次的ADE总和：
        # 1. torch.norm(pred - target, p=2, dim=-1) → 计算每个时间步的L2距离（位移误差）
        #    - pred - target：预测与真实的坐标差（形状[..., H, 2]）
        #    - p=2：计算L2范数（欧氏距离）
        #    - dim=-1：按最后一维（x/y坐标）计算距离，输出形状[..., H]（每个时间步的距离）
        # 2. .mean(dim=-1) → 计算每个样本（智能体）的平均位移误差（所有未来时间步的距离均值）
        #    - dim=-1：按时间步维度（H）求平均，输出形状[..., N]（每个智能体的ADE）
        # 3. .sum() → 计算当前批次所有智能体的ADE总和，累加到self.sum
        self.sum += torch.norm(pred - target, p=2, dim=-1).mean(dim=-1).sum()
        
        # 统计当前批次的样本数量（智能体数）：pred.size(0)为当前批次的智能体数（N），累加到self.count
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        # 计算最终ADE指标：总误差和 ÷ 总样本数（智能体数）
        # 处理count为0的极端情况（避免除以0，虽实际训练中不会出现）
        return self.sum / self.count if self.count != 0 else torch.tensor(0.0)
