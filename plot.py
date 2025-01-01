import matplotlib.pyplot as plt
from helpers import EMA
from icecream import ic 
import numpy as np
import torch

def plot_loss(path_to_save, train=True):
    """
    绘制训练或验证损失曲线
    Args:
        path_to_save: 保存图像的路径
        train: 是否为训练损失(True)或验证损失(False)
    """
    plt.rcParams.update({'font.size': 10})  # 设置字体大小
    with open(path_to_save + "/train_loss.txt", 'r') as f:
        loss_list = [float(line) for line in f.readlines()]  # 读取损失值
    
    title = "Train" if train else "Validation"
    EMA_loss = EMA(loss_list)  # 计算指数移动平均损失
    
    # 绘制原始损失和EMA损失曲线
    plt.plot(loss_list, label = "loss")
    plt.plot(EMA_loss, label="EMA loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title+"_loss")
    plt.savefig(path_to_save+f"/{title}.png")
    plt.close()

def plot_prediction(title, path_to_save, src, tgt, prediction, sensor_number, index_in, index_tar):
    """
    绘制预测结果对比图
    Args:
        title: 图像标题
        path_to_save: 保存路径
        src: 输入序列数据
        tgt: 目标序列数据
        prediction: 预测序列数据
        sensor_number: 传感器编号
        index_in: 输入序列的时间索引
        index_tar: 目标序列的时间索引
    """
    idx_scr = index_in[0, 1:].tolist()  # 输入序列时间点
    idx_tgt = index_tar[0].tolist()     # 目标序列时间点
    idx_pred = [i for i in range(idx_scr[0] +1, idx_tgt[-1])]  # 预测序列时间点

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 16})

    # connect with last elemenet in src
    # tgt = np.append(src[-1], tgt.flatten())
    # prediction = np.append(src[-1], prediction.flatten())

    # plotting
    # 绘制输入、目标和预测序列
    plt.plot(idx_scr, src, '-', color = 'blue', label = 'Input', linewidth=2)
    plt.plot(idx_tgt, tgt, '-', color = 'indigo', label = 'Target', linewidth=2)
    plt.plot(idx_pred, prediction,'--', color = 'limegreen', label = 'Forecast', linewidth=2)

    # formatting
    # 设置图表格式
    plt.grid(visible=True, which='major', linestyle = 'solid')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', linestyle = 'dashed', alpha=0.5)
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.title("Forecast from Sensor " + str(sensor_number[0]))

    # save
    plt.savefig(path_to_save+f"Prediction_{title}.png")
    plt.close()

def plot_training(epoch, path_to_save, src, prediction, sensor_number, index_in, index_tar):
    """
    绘制训练过程中的预测结果
    Args:
        epoch: 当前训练轮次
        path_to_save: 保存路径
        src: 输入序列
        prediction: 预测序列
        sensor_number: 传感器编号
        index_in: 输入序列索引
        index_tar: 目标序列索引
    """
    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))
    # 生成时间索引
    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(visible=True, which='major', linestyle = '-')
    plt.grid(visible=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    # 绘制输入序列和预测序列
    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)

    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()

def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, sensor_number, index_in, index_tar):
    """
    绘制包含采样源的训练过程图
    Args:
        epoch: 当前训练轮次
        path_to_save: 保存路径
        src: 原始输入序列
        sampled_src: 采样后的输入序列
        prediction: 预测序列
        sensor_number: 传感器编号
        index_in: 输入序列索引
        index_tar: 目标序列索引
    注意：使用此函数时需要关闭dropout，否则采样源的绘制可能会受到影响
    """
    # 生成时间索引
    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))
    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]
    idx_sampled_src = [i for i in range(len(sampled_src))]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(visible=True, which='major', linestyle = '-')
    plt.grid(visible=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    ## REMOVE DROPOUT FOR THIS PLOT TO APPEAR AS EXPECTED !! DROPOUT INTERFERES WITH HOW THE SAMPLED SOURCES ARE PLOTTED
    # 绘制采样源、输入序列和预测序列
    plt.plot(idx_sampled_src, sampled_src, 'o-.', color='red', label = 'sampled source', linewidth=1, markersize=10)
    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)
    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()