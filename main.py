import argparse
# from train_teacher_forcing import *
from train_with_sampling import *
from DataLoader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from helpers import *
from inference import *

def main(
    epoch: int = 1000,          # 训练轮数
    k: int = 60,                # transformer模型的维度参数
    batch_size: int = 1,        # 批次大小
    frequency: int = 100,       # 保存模型和打印损失的频率
    training_length = 48,       # 训练序列长度
    forecast_window = 24,       # 预测窗口大小
    train_csv = "train_dataset.csv",  # 训练数据集文件名
    test_csv = "test_dataset.csv",    # 测试数据集文件名
    path_to_save_model = "save_model/",        # 模型保存路径
    path_to_save_loss = "save_loss/",          # 损失记录保存路径
    path_to_save_predictions = "save_predictions/",  # 预测结果保存路径
    device = "cpu"              # 运行设备（CPU/GPU）
):
    """
    主函数：加载数据，训练模型并进行预测
    """
    # 清理输出目录
    clean_directory()

    # 加载训练数据集
    train_dataset = SensorDataset(csv_name = train_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # 加载测试数据集
    test_dataset = SensorDataset(csv_name = test_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 训练模型并获取最佳模型
    best_model = transformer(train_dataloader, epoch, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
    # 使用最佳模型进行预测
    inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000)         # 训练轮数
    parser.add_argument("--k", type=int, default=60)               # transformer维度
    parser.add_argument("--batch_size", type=int, default=1)       # 批次大小
    parser.add_argument("--frequency", type=int, default=100)      # 保存频率
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")      # 模型保存路径
    parser.add_argument("--path_to_save_loss",type=str,default="save_loss/")        # 损失保存路径
    parser.add_argument("--path_to_save_predictions",type=str,default="save_predictions/")  # 预测结果保存路径
    parser.add_argument("--device", type=str, default="cpu")       # 运行设备
    args = parser.parse_args()

    # 使用解析的参数运行主函数
    main(
        epoch=args.epoch,
        k = args.k,
        batch_size=args.batch_size,
        frequency=args.frequency,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        path_to_save_predictions=args.path_to_save_predictions,
        device=args.device,
    )

