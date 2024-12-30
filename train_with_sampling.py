from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math, random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def flip_from_probability(p):
    """根据概率p返回True或False"""
    return True if random.random() < p else False

def transformer(dataloader, EPOCH, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):
    """
    训练Transformer模型的主函数
    
    参数:
        dataloader: 数据加载器
        EPOCH: 训练轮数
        k: 用于计算采样概率的参数
        frequency: 采样频率
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

        # 训练模式 -- TEACHER FORCING
        model.train()
        for index_in, index_tar, _input, target, sensor_number in dataloader:
        
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            optimizer.zero_grad()
            # 调整输入数据维度 [input_length, batch, feature]
            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # [24, 1, 7]
            target = _input.permute(1,0,2).double().to(device)[1:,:,:] # 将src向后移动一位作为目标
            sampled_src = src[:1, :, :] # 取第一个时间步作为起始点 [1, 1, 7]

            for i in range(len(target)-1):
                # 使用当前序列进行预测
                prediction = model(sampled_src, device) # torch.Size([1xw, 1, 1])
                # for p1, p2 in zip(params, model.parameters()):
                #     if p1.data.ne(p2.data).sum() > 0:
                #         ic(False)
                # ic(True)
                # ic(i, sampled_src[:,:,0], prediction)
                # time.sleep(1)
                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """

                # 在训练初期(前24步)使用真实值
                if i < 24: # 一天的数据量，足够推断周期性
                    prob_true_val = True
                else:
                    # 根据epoch计算采样概率
                    v = k/(k+math.exp(epoch/k)) # 概率随epoch变化
                    prob_true_val = flip_from_probability(v) # 在epoch 0时有超过95%的概率使用真实值

                if prob_true_val: # 使用真实值作为下一个输入
                    sampled_src = torch.cat((sampled_src.detach(), src[i+1, :, :].unsqueeze(0).detach()))
                else: # 使用预测值作为下一个输入
                    positional_encodings_new_val = src[i+1,:,1:].unsqueeze(0)
                    predicted_humidity = torch.cat((prediction[-1,:,:].unsqueeze(0), positional_encodings_new_val), dim=2)
                    sampled_src = torch.cat((sampled_src.detach(), predicted_humidity.detach()))
            
            # 计算整个序列的损失并更新模型
            loss = criterion(target[:-1,:,0].unsqueeze(-1), prediction)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        # 保存最佳模型
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"

        # 每10个epoch绘制单步预测结果
        if epoch % 10 == 0:
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            # 反归一化数据用于绘图
            scaler = load('scalar_item.joblib')
            sampled_src_humidity = scaler.inverse_transform(sampled_src[:,:,0].cpu()) #torch.Size([35, 1, 7])
            src_humidity = scaler.inverse_transform(src[:,:,0].cpu()) #torch.Size([35, 1, 7])
            target_humidity = scaler.inverse_transform(target[:,:,0].cpu()) #torch.Size([35, 1, 7])
            prediction_humidity = scaler.inverse_transform(prediction[:,:,0].detach().cpu().numpy()) #torch.Size([35, 1, 7])
            plot_training_3(epoch, path_to_save_predictions, src_humidity, sampled_src_humidity, prediction_humidity, sensor_number, index_in, index_tar)

        # 计算平均训练损失并记录
        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    # 绘制训练过程中的损失变化
    plot_loss(path_to_save_loss, train=True)
    return best_model