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


class MR(Metric):
    # MR（Miss Rate，缺失率）评价指标
    # 用于量化轨迹预测的"失效比例"：统计预测轨迹终点与真实终点的距离超过阈值的智能体占比
    # 继承自PyTorch Lightning的Metric类，支持分布式训练（多GPU）下的指标同步，确保跨设备统计准确

    def __init__(self,
                 miss_threshold: float = 2.0,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        # 初始化父类Metric：配置指标计算时机、分布式参数及自定义阈值
        super(MR, self).__init__(
            compute_on_step=compute_on_step,  # 是否在每个训练/验证step实时计算指标（默认True）
            dist_sync_on_step=dist_sync_on_step,  # 是否在每个step同步分布式指标（默认False，减少通信开销）
            process_group=process_group,  # 分布式训练的进程组（可选，默认用全局进程组）
            dist_sync_fn=dist_sync_fn  # 自定义分布式同步函数（可选，默认用框架自带逻辑）
        )
        
        # 注册状态变量：用于累积统计的中间结果（支持分布式同步时按"求和"合并）
        # sum：累积"预测失效"的智能体数量（初始为0.0，多GPU训练时汇总各卡的失效数）
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        # count：累积总智能体数量（初始为0，多GPU训练时汇总各卡的总样本数）
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        # 预测失效的距离阈值（默认2.0米，符合自动驾驶运动预测的行业惯例）
        # 若预测终点与真实终点的距离>该阈值，判定为"失效"
        self.miss_threshold = miss_threshold

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        # 更新指标状态：每批数据统计当前批次的失效智能体数，累加到sum和count
        # pred：模型预测的轨迹，形状通常为[F, N, H, 2]或[N, H, 2]（F=模态数，N=智能体数，H=未来步，2=x/y坐标）
        # target：真实轨迹，形状通常为[N, H, 2]（与pred的位置维度对应）
        
        # 分步统计当前批次的失效智能体数：
        # 1. pred[:, -1] / target[:, -1] → 提取"最后一个未来时间步"的预测/真实坐标（关注终点）
        #    - 索引[:, -1]：对时间步维度（H）取最后一个元素，形状从[..., H, 2]→[..., 2]
        # 2. torch.norm(..., p=2, dim=-1) → 计算终点的L2距离（欧氏距离），形状[..., N]
        # 3. > self.miss_threshold → 判定是否失效：距离>阈值为True（失效），否则为False，形状[..., N]
        # 4. .sum() → 统计当前批次失效的智能体总数（True按1计数，False按0计数），累加到self.sum
        self.sum += (torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1) > self.miss_threshold).sum()
        
        # 统计当前批次的总智能体数：pred.size(0)为当前批次的智能体数（N），累加到self.count
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        # 计算最终MR指标：失效智能体数 ÷ 总智能体数（即失效比例，值越小表示预测可靠性越高）
        # 避免count=0时除以0的极端情况（实际训练/验证中不会出现）
        return self.sum / self.count if self.count != 0 else torch.tensor(0.0)
