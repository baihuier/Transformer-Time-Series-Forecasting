import os, shutil

# save train or validation loss
# 保存训练或验证损失值
def log_loss(loss_val : float, path_to_save_loss : str, train : bool = True):
    # 根据train参数决定保存的文件名
    if train:
        file_name = "train_loss.txt"
    else:
        file_name = "val_loss.txt"

    # 构建完整的文件路径并确保目录存在
    path_to_file = path_to_save_loss+file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    # 以追加模式打开文件并写入损失值
    with open(path_to_file, "a") as f:
        f.write(str(loss_val)+"\n")
        f.close()

# Exponential Moving Average, https://en.wikipedia.org/wiki/Moving_average
# 指数移动平均，用于平滑数据序列
def EMA(values, alpha=0.1):
    # 初始化EMA列表，使用第一个值作为起始点
    ema_values = [values[0]]
    # 计算后续值的EMA
    for idx, item in enumerate(values[1:]):
        ema_values.append(alpha*item + (1-alpha)*ema_values[idx])
    return ema_values

# Remove all files from previous executions and re-run the model.
# 清理之前执行产生的文件并重新创建必要的目录
def clean_directory():
    # 如果存在则删除这些目录
    if os.path.exists('save_loss'):
        shutil.rmtree('save_loss')
    if os.path.exists('save_model'): 
        shutil.rmtree('save_model')
    if os.path.exists('save_predictions'): 
        shutil.rmtree('save_predictions')
    # 创建新的空目录
    os.mkdir("save_loss")
    os.mkdir("save_model")
    os.mkdir("save_predictions")