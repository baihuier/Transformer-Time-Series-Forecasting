import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic

class SensorDataset(Dataset):
    """传感器数据集类，用于加载和预处理时间序列数据"""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        初始化数据集
        Args:
            csv_name (string): CSV文件名
            root_dir (string): 根目录路径
            training_length (int): 训练序列长度
            forecast_window (int): 预测窗口大小
        """
        
        # 加载原始数据文件
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()  # 初始化MinMax标准化器
        self.T = training_length        # 训练序列长度
        self.S = forecast_window        # 预测窗口大小

    def __len__(self):
        """返回数据集中传感器的数量"""
        return len(self.df.groupby(by=["reindexed_id"]))
    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        """
        获取单个数据样本
        Args:
            idx (int): 传感器索引
        Returns:
            index_in: 输入序列的时间索引
            index_tar: 目标序列的时间索引
            _input: 标准化后的输入特征序列
            target: 标准化后的目标特征序列
            sensor_number: 传感器编号
        """
        
        # 传感器索引从1开始
        idx = idx+1

        # 随机选择时间窗口的起始位置
        # np.random.seed(0)
        start = np.random.randint(0, len(self.df[self.df["reindexed_id"]==idx]) - self.T - self.S) 
        
        # 获取传感器编号
        sensor_number = str(self.df[self.df["reindexed_id"]==idx][["sensor_id"]][start:start+1].values.item())
        
        # 生成输入和目标序列的时间索引
        index_in = torch.tensor([i for i in range(start, start+self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        
        # 提取特征序列：湿度和时间编码（小时、天、月的正弦和余弦值）
        _input = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + self.T].values)
        target = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start + self.T : start + self.T + self.S].values)

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        # 仅对湿度值进行标准化，时间编码已经在[-1,1]范围内
        scaler = self.transform

        # 仅使用输入数据拟合scaler，避免数据泄露
        scaler.fit(_input[:,0].unsqueeze(-1))
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        # 保存scaler用于后续反向转换数据进行可视化
        dump(scaler, 'scalar_item.joblib')

        return index_in, index_tar, _input, target, sensor_number