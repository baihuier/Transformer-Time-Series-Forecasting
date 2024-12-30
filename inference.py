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

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):
    """
    执行模型推理
    Args:
        path_to_save_predictions: 保存预测结果的路径
        forecast_window: 预测窗口大小
        dataloader: 数据加载器
        device: 计算设备(CPU/GPU)
        path_to_save_model: 模型保存路径
        best_model: 最佳模型文件名
    """
    device = torch.device(device)
    
    # 加载模型
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(path_to_save_model+best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad():
        model.eval()
        # 生成25个预测图
        for plot in range(25):
            for index_in, index_tar, _input, target, sensor_number in dataloader:
                
                # 准备输入数据
                # 从1开始以使src与target匹配，但保持与训练时相同的长度
                src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 形状: 47, 1, 7 (t1到t47)
                target = target.permute(1,0,2).double().to(device) # 形状: t48到t59

                next_input_model = src
                all_predictions = []

                # 循环生成预测
                for i in range(forecast_window - 1):
                    # 使用模型进行预测
                    prediction = model(next_input_model, device) # 形状: 47,1,1 (t2'到t48')

                    # 收集预测结果
                    if all_predictions == []:
                        all_predictions = prediction # 47,1,1: t2' - t48'
                    else:
                        all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                    # 更新位置编码
                    pos_encoding_old_vals = src[i+1:, :, 1:] # 46, 1, 6, 获取第一个位置编码值: t2到t47
                    pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1) # 1, 1, 6, 添加最后预测值的位置编码: t48
                    pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 与预测匹配的位置编码: t2到t48
                    
                    # 准备下一轮输入
                    next_input_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) # t2到t47, t48'
                    next_input_model = torch.cat((next_input_model, pos_encodings), dim = 2) # 47, 1, 7 下一轮的输入

                # 计算损失
                true = torch.cat((src[1:,:,0],target[:-1,:,0]))
                loss = criterion(true, all_predictions[:,:,0])
                val_loss += loss
            
            # 反归一化处理并绘制预测结果
            val_loss = val_loss/10
            scaler = load('scalar_item.joblib')
            src_humidity = scaler.inverse_transform(src[:,:,0].cpu())
            target_humidity = scaler.inverse_transform(target[:,:,0].cpu())
            prediction_humidity = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())
            plot_prediction(plot, path_to_save_predictions, src_humidity, target_humidity, prediction_humidity, sensor_number, index_in, index_tar)

        # 输出在未见数据集上的损失
        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")