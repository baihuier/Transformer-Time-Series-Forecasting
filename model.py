import torch.nn as nn
import torch, math
from icecream import ic
import time

"""
此模型架构基于论文 "Attention Is All You Need"。
作者: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, 
      Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""

class Transformer(nn.Module):
    """
    用于时间序列预测的Transformer模型
    
    参数:
        feature_size (int): 输入特征的维度
        num_layers (int): Transformer编码器层的数量
        dropout (float): dropout比率，用于防止过拟合
    """
    def __init__(self, feature_size=7, num_layers=3, dropout=0):
        super(Transformer, self).__init__()
        
        # 创建一个Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,  # 模型的维度
            nhead=7,              # 注意力头的数量
            dropout=dropout
        )
        
        # 创建完整的Transformer编码器，包含多个编码器层
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        
        # 最后的线性层，用于将特征映射到预测值
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        """初始化解码器的权重和偏置"""
        initrange = 0.1    
        self.decoder.bias.data.zero_()  # 将偏置初始化为0
        self.decoder.weight.data.uniform_(-initrange, initrange)  # 将权重初始化为均匀分布

    def _generate_square_subsequent_mask(self, sz):
        """
        生成一个上三角掩码矩阵，用于屏蔽未来时间步的信息
        
        参数:
            sz (int): 序列长度
            
        返回:
            mask (Tensor): 掩码矩阵，其中未来时间步的位置被设置为负无穷
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device):
        """
        前向传播函数
        
        参数:
            src (Tensor): 输入序列
            device (torch.device): 计算设备(CPU/GPU)
            
        返回:
            output (Tensor): 模型的预测输出
        """
        # 生成掩码并将其移动到指定设备
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        # 通过Transformer编码器处理输入
        output = self.transformer_encoder(src, mask)
        # 通过解码器生成预测
        output = self.decoder(output)
        return output
        

