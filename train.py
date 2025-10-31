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
from argparse import ArgumentParser  # 用于解析命令行参数

import pytorch_lightning as pl  # 简化PyTorch训练流程的高级框架
from pytorch_lightning.callbacks import ModelCheckpoint  # 模型保存回调工具

from datamodules import ArgoverseV1DataModule  # Argoverse数据集的数据模块
from models.hivt import HiVT  # HiVT模型定义

if __name__ == '__main__':
    # 固定所有随机种子，保证实验结果可复现（论文中实验的一致性基础）
    pl.seed_everything(2022)

    # 创建命令行参数解析器
    parser = ArgumentParser()
    
    # 添加数据相关的参数
    # 数据集根目录（必须通过命令行指定，如--root /path/to/argoverse）
    parser.add_argument('--root', type=str, required=True)
    # 训练批次大小（论文中设置为32，平衡效率与训练稳定性）
    parser.add_argument('--train_batch_size', type=int, default=32)
    # 验证批次大小（与训练批次保持一致，便于GPU内存利用）
    parser.add_argument('--val_batch_size', type=int, default=32)
    # 训练集是否打乱顺序（True表示打乱，增强模型泛化性）
    parser.add_argument('--shuffle', type=bool, default=True)
    # 数据加载的线程数（8线程加速数据读取，避免GPU等待）
    parser.add_argument('--num_workers', type=int, default=8)
    # 是否锁定内存（GPU训练时启用，加速数据从内存到GPU的传输）
    parser.add_argument('--pin_memory', type=bool, default=True)
    # 是否保持数据加载线程存活（True表示不重复创建线程，减少开销）
    parser.add_argument('--persistent_workers', type=bool, default=True)
    
    # 添加训练相关的参数
    # 使用的GPU数量（默认1卡训练，论文实验配置）
    parser.add_argument('--gpus', type=int, default=1)
    # 最大训练轮次（论文中设置为64轮，经实验验证的合理轮次）
    parser.add_argument('--max_epochs', type=int, default=64)
    # 验证时监控的指标（默认跟踪val_minFDE，即最小最终位移误差）
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    # 保存表现前k名的模型（默认保存前5个最优模型）
    parser.add_argument('--save_top_k', type=int, default=5)
    
    # 向解析器添加HiVT模型特有的参数（如隐藏层维度、注意力头数等）
    parser = HiVT.add_model_specific_args(parser)
    # 解析所有命令行参数，得到参数对象
    args = parser.parse_args()

    # 初始化模型保存回调：根据monitor指标保存最优模型，mode='min'表示指标越小越好
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    
    # 初始化训练器：基于解析的参数配置，整合模型保存回调
    # pl.Trainer自动处理训练循环、GPU调度、日志等底层逻辑
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])
    
    # 初始化HiVT模型：将参数转为字典传入，配置模型结构与超参数
    model = HiVT(**vars(args))
    
    # 初始化数据模块：基于命令行参数配置数据加载逻辑
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    
    # 启动训练：自动执行训练循环（training_step）和验证循环（validation_step）
    trainer.fit(model, datamodule)

    '''
    trainer.fit：启动训练流程，内部逻辑为：
        1. 从datamodule获取训练集和验证集数据加载器。
        2. 循环max_epochs轮，每轮先执行model.training_step（训练步骤，计算损失、反向传播）。
        3. 每轮结束后执行model.validation_step（验证步骤，计算 minADE、minFDE 等指标）。
        4. 根据验证指标更新ModelCheckpoint，保存最优模型。
    '''


    '''
    # 在原来的代码中修改ModelCheckpoint的初始化
    model_checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode='min',
        dirpath='./hivt_checkpoints/'  # 自定义保存目录
    )


    # Saving checkpoint to: lightning_logs/version_0/checkpoints/epoch=10-val_minFDE=0.98.ckpt
    '''
