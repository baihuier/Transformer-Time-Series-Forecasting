from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time # 用于调试
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def transformer(dataloader, EPOCH, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):
    """
    使用教师强制(teacher forcing)方法训练Transformer模型
    
    参数:
        dataloader: 数据加载器
        EPOCH: 训练轮数
        frequency: 频率参数
        path_to_save_model: 模型保存路径
        path_to_save_loss: 损失值保存路径
        path_to_save_predictions: 预测结果保存路径
        device: 训练设备(CPU/GPU)
    """
    device = torch.device(device)

    # 初始化模型、优化器和损失函数
    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        val_loss = 0

        # 训练模式：Teacher Forcing
        model.train()
        for index_in, index_tar, _input, target, sensor_number in dataloader: # 遍历每个数据集
            optimizer.zero_grad()

            # 输入数据形状调整
            # _input形状: [batch, input_length, feature]
            # 模型所需输入形状: [input_length, batch, feature]
            
            # 准备源序列和目标序列
            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # 形状: [24, 1, 7]
            target = _input.permute(1,0,2).double().to(device)[1:,:,:] # 源序列向后移动一位作为目标
            
            # 前向传播
            prediction = model(src, device) # 形状: [24, 1, 7]
            
            # 计算损失
            loss = criterion(prediction, target[:,:,0].unsqueeze(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.detach().item())
            train_loss += loss.detach().item()

        # 保存最佳模型
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"

        # 每100轮绘制一次单步预测结果
        if epoch % 100 == 0:
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            
            # 数据反归一化处理
            scaler = load('scalar_item.joblib')
            src_humidity = scaler.inverse_transform(src[:,:,0].cpu()) #torch.Size([35, 1, 7])
            target_humidity = scaler.inverse_transform(target[:,:,0].cpu()) #torch.Size([35, 1, 7])
            prediction_humidity = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy()) #torch.Size([35, 1, 7])
            # 绘制训练过程图
            plot_training(epoch, path_to_save_predictions, src_humidity, prediction_humidity, sensor_number, index_in, index_tar)

        # 计算平均训练损失并记录
        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
    
    # 绘制损失曲线    
    plot_loss(path_to_save_loss, train=True)
    return best_model