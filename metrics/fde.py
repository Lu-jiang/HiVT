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


class FDE(Metric):
    # FDE（Final Displacement Error，最终位移误差）评价指标
    # 用于量化轨迹预测的终点精度：仅计算预测轨迹与真实轨迹在"最后一个未来时间步"的L2距离
    # 继承自PyTorch Lightning的Metric类，支持分布式训练（多GPU）下的指标同步，确保结果准确聚合

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        # 初始化父类Metric：配置指标计算时机和分布式参数
        super(FDE, self).__init__(
            compute_on_step=compute_on_step,  # 是否在每个训练/验证step实时计算指标（默认True）
            dist_sync_on_step=dist_sync_on_step,  # 是否在每个step同步分布式指标（默认False，减少通信开销）
            process_group=process_group,  # 分布式训练的进程组（可选，默认用全局进程组）
            dist_sync_fn=dist_sync_fn  # 自定义分布式同步函数（可选，默认用框架自带逻辑）
        )
        
        # 注册状态变量：用于累积计算的中间结果（支持分布式同步时按"求和"合并）
        # sum：累积所有样本的FDE总和（初始为0.0，多GPU训练时会自动汇总各卡的sum）
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        # count：累积样本数量（即智能体总数，初始为0，多GPU训练时汇总各卡的count）
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        # 更新指标状态：每批数据计算当前批次的FDE贡献，累加到sum和count
        # pred：模型预测的轨迹，形状通常为[F, N, H, 2]或[N, H, 2]（F=模态数，N=智能体数，H=未来步，2=x/y坐标）
        # target：真实轨迹，形状通常为[N, H, 2]（与pred的位置维度对应，仅含真实坐标）
        
        # 分步计算当前批次的FDE总和：
        # 1. pred[:, -1] / target[:, -1] → 提取"最后一个未来时间步"的预测/真实坐标
        #    - 索引[:, -1]：对第2维（时间步维度H）取最后一个元素，形状从[..., H, 2]→[..., 2]
        # 2. torch.norm(..., p=2, dim=-1) → 计算最后一步的L2距离（欧氏距离，即FDE）
        #    - p=2：指定L2范数，dim=-1：按x/y坐标维度计算距离，输出形状[..., N]（每个智能体的FDE）
        # 3. .sum() → 计算当前批次所有智能体的FDE总和，累加到self.sum
        self.sum += torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1).sum()
        
        # 统计当前批次的智能体数量：pred.size(0)为当前批次的智能体数（N），累加到self.count
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        # 计算最终FDE指标：总FDE误差和 ÷ 总智能体数（避免count=0时除以0的极端情况）
        return self.sum / self.count if self.count != 0 else torch.tensor(0.0)
