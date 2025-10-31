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
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from utils import TemporalData


class HiVT(pl.LightningModule):
    # HiVT模型的核心类，继承自PyTorch Lightning的LightningModule
    # 封装了模型结构、前向传播、损失计算、优化器配置等完整逻辑

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,** kwargs) -> None:
        # 初始化父类（LightningModule）
        super(HiVT, self).__init__()
        
        # 保存所有超参数到self.hparams，方便后续查看和加载
        # 例如self.hparams.embed_dim可获取嵌入维度
        self.save_hyperparameters()
        
        # 将关键参数绑定为实例变量，方便在类内其他方法中调用
        self.historical_steps = historical_steps  # 历史时间步数
        self.future_steps = future_steps          # 未来预测步数
        self.num_modes = num_modes                # 多模态数量
        self.rotate = rotate                      # 是否旋转变换
        self.parallel = parallel                  # 是否并行计算局部编码
        self.lr = lr                              # 学习率
        self.weight_decay = weight_decay          # 权重衰减系数
        self.T_max = T_max                        # 余弦退火周期
        
        # 初始化局部编码器（Local Encoder）
        # 负责提取单个智能体为中心的局部特征（智能体交互、时间依赖、车道交互）
        self.local_encoder = LocalEncoder(
            historical_steps=historical_steps,  # 历史时间步数
            node_dim=node_dim,                  # 智能体轨迹向量维度
            edge_dim=edge_dim,                  # 关系向量维度
            embed_dim=embed_dim,                # 嵌入维度
            num_heads=num_heads,                # 注意力头数
            dropout=dropout,                    # Dropout概率
            num_temporal_layers=num_temporal_layers,  # 时间编码器层数
            local_radius=local_radius,          # 局部区域半径
            parallel=parallel                   # 是否并行计算
        )
        
        # 初始化全局交互模块（Global Interactor）
        # 负责捕捉跨局部区域的长距离依赖，更新局部特征为全局特征
        self.global_interactor = GlobalInteractor(
            historical_steps=historical_steps,  # 历史时间步数
            embed_dim=embed_dim,                # 嵌入维度
            edge_dim=edge_dim,                  # 关系向量维度
            num_modes=num_modes,                # 多模态数量
            num_heads=num_heads,                # 注意力头数
            num_layers=num_global_layers,       # 全局Transformer层数
            dropout=dropout,                    # Dropout概率
            rotate=rotate                       # 是否旋转变换（用于跨区域坐标对齐）
        )
        
        # 初始化解码器（MLPDecoder）
        # 融合局部和全局特征，输出多模态轨迹预测
        self.decoder = MLPDecoder(
            local_channels=embed_dim,           # 局部特征维度
            global_channels=embed_dim,          # 全局特征维度
            future_steps=future_steps,          # 未来预测步数
            num_modes=num_modes,                # 多模态数量
            uncertain=True                      # 是否预测轨迹不确定性（输出均值+尺度）
        )
        
        # 初始化回归损失函数：拉普拉斯负对数似然损失
        # 用于计算预测轨迹与真实轨迹的误差（考虑不确定性）
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        
        # 初始化分类损失函数：软目标交叉熵损失
        # 用于优化多模态轨迹的概率权重
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        
        # 初始化评价指标计算器
        self.minADE = ADE()    # 最小平均位移误差计算器
        self.minFDE = FDE()    # 最小最终位移误差计算器
        self.minMR = MR()      # 缺失率计算器（终点误差>2米的比例）

    def forward(self, data: TemporalData):
        # HiVT模型的前向传播核心逻辑：串联局部编码、全局交互、轨迹预测三个核心模块
        # 输入：TemporalData实例（包含智能体轨迹、车道特征、图结构等所有预处理数据）
        # 输出：预测结果（y_hat：多模态轨迹，pi：模态概率）

        # 1. 旋转增强（可选）：将智能体轨迹旋转到自身局部坐标系，增强模型对方向变化的鲁棒性
        if self.rotate:  # 若启用旋转旋转增强（初始化时配置）
            # 1.1 构建每个智能体的旋转矩阵（基于自身运动方向角度rotate_angles）
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)  # 形状[N, 2, 2]，N=智能体数
            sin_vals = torch.sin(data['rotate_angles'])  # 角度正弦值（形状[N]）
            cos_vals = torch.cos(data['rotate_angles'])  # 角度余弦值（形状[N]）
            # 填充旋转矩阵（2D旋转公式：[cosθ, -sinθ; sinθ, cosθ]）
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            
            # 1.2 旋转真实轨迹标签（若存在）：与预测值的坐标系保持一致（仅训练/验证时需要）
            if data.y is not None:  # data.y为未来轨迹标签（测试集无标签）
                # 批量矩阵乘法：将每个智能体的未来轨迹（[N, F, 2]）与自身旋转矩阵相乘
                data.y = torch.bmm(data.y, rotate_mat)  # 形状[N, F, 2] → 旋转后的标签
            
            # 1.3 保存旋转矩阵到数据中（供后续模块使用，如局部编码器）
            data['rotate_mat'] = rotate_mat
        else:
            # 不启用旋转增强时，旋转矩阵设为None
            data['rotate_mat'] = None

        # 2. 局部编码：提取每个智能体的时序特征和局部交互特征（含智能体-车道交互）
        # 输入：完整的TemporalData（含x、lane_vectors、lane_actor_index等）
        # 输出：local_embed（局部嵌入特征），形状[N, L]（L=局部特征维度）
        local_embed = self.local_encoder(data=data)

        # 3. 全局交互：建模智能体之间的长距离依赖关系，聚合全局场景信息
        # 输入：TemporalData + 局部嵌入特征local_embed
        # 输出：global_embed（全局嵌入特征），形状[F, N, G]（F=模态数，N=智能体数，G=全局特征维度）
        global_embed = self.global_interactor(data=data, local_embed=local_embed)

        # 4. 多模态解码：基于局部+全局特征，预测未来轨迹的多模态分布
        # 输入：local_embed（局部特征） + global_embed（全局特征）
        # 输出：
        #   y_hat：预测轨迹（含位置+不确定性），形状[F, N, H, 4]（H=未来步，4=x/y+scale_x/scale_y）
        #   pi：模态概率，形状[N, F]（每个智能体对应各模态的权重）
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)

        # 返回预测结果（用于计算损失或输出最终预测）
        return y_hat, pi

    def training_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    # def configure_optimizers(self):
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
    #     blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
    #     for module_name, module in self.named_modules():
    #         for param_name, param in module.named_parameters():
    #             full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
    #             if 'bias' in param_name:
    #                 no_decay.add(full_param_name)
    #             elif 'weight' in param_name:
    #                 if isinstance(module, whitelist_weight_modules):
    #                     decay.add(full_param_name)
    #                 elif isinstance(module, blacklist_weight_modules):
    #                     no_decay.add(full_param_name)
    #             elif not ('weight' in param_name or 'bias' in param_name):
    #                 no_decay.add(full_param_name)
    #     param_dict = {param_name: param for param_name, param in self.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     assert len(inter_params) == 0
    #     assert len(param_dict.keys() - union_params) == 0

    #     optim_groups = [
    #         {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
    #          "weight_decay": self.weight_decay},
    #         {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
    #          "weight_decay": 0.0},
    #     ]

    #     optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        
        # 1. 按模块类型划分需要/不需要权重衰减的参数（你的原有逻辑完全保留）
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "参数不能同时属于 decay 和 no_decay"
        assert len(param_dict.keys() - union_params) == 0, "存在未分类的参数"

        # 2. 构建优化器参数组（原有逻辑保留）
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
            "weight_decay": self.weight_decay},  # 需要权重衰减的参数组
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
            "weight_decay": 0.0}  # 不需要权重衰减的参数组（如BN、偏置）
        ]

        # 3. 定义优化器和调度器（原有逻辑保留）
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=self.T_max,  # 学习率从峰值降到最小值的周期（需确保 self.T_max 已在 __init__ 中定义）
            eta_min=0.0        # 学习率下限
        )

        # 4. 关键修改：按 PL 要求返回「优化器 + 带配置的调度器」
        # 格式：返回字典，包含 optimizer 和 lr_scheduler（调度器对象 + 配置字典）
        return {
            "optimizer": optimizer,  # 优化器（单优化器直接传对象，多优化器传列表）
            "lr_scheduler": {
                "scheduler": scheduler,  # 调度器对象
                "interval": "epoch",     # 更新频率："epoch"（每个epoch更新）或 "step"（每个batch更新）
                "frequency": 1,          # 每 1 个 interval 更新一次（默认1，无需修改）
                # "monitor": "val_loss"  # 仅用于「基于指标的调度器」（如 ReduceLROnPlateau），CosineAnnealingLR 无需此参数
            }
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        # 为HiVT模型创建独立的参数组，避免与其他参数混淆
        parser = parent_parser.add_argument_group('HiVT')
        
        # 历史观测时间步数（单位：步）
        # 论文中设置为20步，对应2秒（每步0.1秒），与Argoverse数据集的观测周期一致
        parser.add_argument('--historical_steps', type=int, default=20)
        
        # 未来预测时间步数（单位：步）
        # 论文中设置为30步，对应3秒（每步0.1秒），是自动驾驶场景的标准预测周期
        parser.add_argument('--future_steps', type=int, default=30)
        
        # 多模态预测的轨迹模式数量
        # 论文中设置为6种模式，通过混合拉普拉斯分布覆盖不同可能的未来轨迹
        parser.add_argument('--num_modes', type=int, default=6)
        
        # 是否对局部区域进行旋转变换
        # 默认为True，实现旋转不变性：以中心智能体运动方向为基准旋转所有局部向量
        parser.add_argument('--rotate', type=bool, default=True)
        
        # 智能体轨迹向量的原始维度
        # 2D场景中为x、y方向的坐标差，故维度为2（对应论文中的轨迹向量表示）
        parser.add_argument('--node_dim', type=int, default=2)
        
        # 智能体间/智能体-车道间关系向量的原始维度
        # 2D场景中为x、y方向的相对位置，故维度为2（对应论文中的相对位置向量）
        parser.add_argument('--edge_dim', type=int, default=2)
        
        # 特征嵌入维度（模型隐藏层维度）
        # 论文中有HiVT-64（64）和HiVT-128（128）两种配置，必须通过命令行指定
        parser.add_argument('--embed_dim', type=int, required=True)
        
        # 多头注意力的头数
        # 论文中设置为8头，用于并行捕捉不同子空间的特征交互
        parser.add_argument('--num_heads', type=int, default=8)
        
        # Dropout概率（防止过拟合）
        # 论文中设置为0.1，在注意力层和MLP层后应用
        parser.add_argument('--dropout', type=float, default=0.1)
        
        # 时间编码器（Temporal Encoder）的Transformer层数
        # 论文中设置为4层，用于捕捉历史轨迹的时间依赖关系
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        
        # 全局交互模块（Global Interactor）的Transformer层数
        # 论文中设置为3层，用于捕捉跨局部区域的长距离依赖
        parser.add_argument('--num_global_layers', type=int, default=3)
        
        # 局部编码器的区域半径（单位：米）
        # 论文中默认50米，控制局部区域内智能体和车道的筛选范围
        parser.add_argument('--local_radius', type=float, default=50)
        
        # 是否并行计算多个智能体的局部编码
        # 默认为False，若开启可加速训练（需硬件支持），不影响模型结构
        parser.add_argument('--parallel', type=bool, default=False)
        
        # 初始学习率
        # 论文中设置为5e-4，用于Adam优化器
        parser.add_argument('--lr', type=float, default=5e-4)
        
        # 权重衰减系数（L2正则化）
        # 论文中设置为1e-4，防止模型过拟合
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        
        # 余弦退火学习率调度的周期（总迭代轮次）
        # 与max_epochs保持一致（64），实现学习率随训练轮次余弦衰减
        parser.add_argument('--T_max', type=int, default=64)
        
        # 返回添加完模型参数的解析器
        return parent_parser
