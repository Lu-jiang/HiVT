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
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset


'''
与数据集类的关联
ArgoverseV1DataModule依赖ArgoverseV1Dataset：
    通过初始化ArgoverseV1Dataset实例，获取预处理后的样本；
数据流向：
    ArgoverseV1Dataset（单样本预处理）→ DataLoader（批处理、并行加载）→ 模型输入（训练 / 验证）。
'''
class ArgoverseV1DataModule(LightningDataModule):
    # Argoverse V1数据模块：基于PyTorch Lightning的LightningDataModule实现
    # 核心作用是"统一管理数据流程"，将数据集加载、预处理、批处理逻辑与模型训练代码解耦
    # 自动适配训练/验证阶段的不同数据需求（如训练 shuffle、验证不 shuffle），支持分布式训练

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        # 初始化数据模块：配置数据路径、批处理参数、数据增强等核心参数
        super(ArgoverseV1DataModule, self).__init__()
        
        # 1. 数据集基础配置
        self.root = root  # 数据集根目录（与ArgoverseV1Dataset的root一致）
        self.local_radius = local_radius  # 局部区域半径（传递给数据集，确保与模型匹配）
        
        # 2. 批处理参数（训练/验证可独立设置，适配不同阶段的内存需求）
        self.train_batch_size = train_batch_size  # 训练集批次大小（如32）
        self.val_batch_size = val_batch_size      # 验证集批次大小（如32，可与训练一致）
        self.shuffle = shuffle                    # 训练集是否打乱样本（默认True，增强泛化性）
        
        # 3. 数据加载优化参数（提升加载速度，减少GPU等待）
        self.num_workers = num_workers                # 数据加载线程数（如8，并行读取数据）
        self.pin_memory = pin_memory                  # 是否锁定内存（GPU训练时启用，加速数据传输）
        self.persistent_workers = persistent_workers  # 是否保持加载线程存活（避免反复创建线程，减少开销）
        
        # 4. 数据增强（训练/验证可独立设置，验证集通常不增强）
        self.train_transform = train_transform  # 训练集数据增强（如随机旋转、加噪声）
        self.val_transform = val_transform      # 验证集数据增强（默认None，保持数据原始性）

    def prepare_data(self) -> None:
        # 数据准备阶段：仅在主进程执行，用于触发数据集的原始数据下载、预处理（非必须，视数据集实现而定）
        # 此处初始化训练/验证数据集，若ArgoverseV1Dataset的__init__包含自动下载/预处理逻辑，会在此触发
        # 注意：不返回数据集，仅执行"准备动作"（如创建预处理目录、下载压缩包）
        ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.local_radius)
        ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        # 数据集初始化阶段：在每个进程（分布式训练时）执行，创建训练/验证数据集实例
        # stage：可选参数，指定当前阶段（'fit'训练阶段、'test'测试阶段），此处仅处理训练/验证
        # 作用：避免分布式训练时多个进程重复初始化数据集，确保每个进程有独立的数据集实例
        
        # 初始化训练集：传入训练阶段的参数（split='train'、训练数据增强）
        self.train_dataset = ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.local_radius)
        # 初始化验证集：传入验证阶段的参数（split='val'、验证数据增强）
        self.val_dataset = ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.local_radius)

    def train_dataloader(self):
        # 返回训练集数据加载器（DataLoader）：模型训练时会自动调用此方法获取训练批次
        return DataLoader(
            dataset=self.train_dataset,          # 训练数据集实例
            batch_size=self.train_batch_size,    # 训练批次大小
            shuffle=self.shuffle,                # 打乱样本（训练集专用）
            num_workers=self.num_workers,        # 加载线程数
            pin_memory=self.pin_memory,          # 锁定内存（加速GPU传输）
            persistent_workers=self.persistent_workers  # 保持线程存活
        )

    def val_dataloader(self):
        # 返回验证集数据加载器（DataLoader）：模型验证时会自动调用此方法获取验证批次
        return DataLoader(
            dataset=self.val_dataset,            # 验证数据集实例
            batch_size=self.val_batch_size,      # 验证批次大小
            shuffle=False,                       # 不打乱样本（验证集需固定顺序，确保结果可复现）
            num_workers=self.num_workers,        # 加载线程数（与训练一致，充分利用资源）
            pin_memory=self.pin_memory,          # 锁定内存
            persistent_workers=self.persistent_workers  # 保持线程存活
        )