"""
Flow Matching for Channel Gain Map Generation  # 英文标题：用于生成信道增益图的流匹配训练脚本
基于 Flow Matching 的信道增益图生成训练脚本  # 中文标题：说明脚本功能
"""

import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.utils.data import Dataset, DataLoader  # 导入数据集与数据加载器
import numpy as np  # 导入 NumPy 数值计算库
import os  # 导入操作系统路径与文件操作库
import glob  # 导入通配符文件查找库
import matplotlib.pyplot as plt  # 导入 Matplotlib 绘图库
from tqdm import tqdm  # 导入进度条库
import json  # 导入 JSON 处理库（当前未使用，保留扩展）
from PIL import Image  # 导入图像读取与处理库 Pillow
from torchvision import transforms  # 导入图像张量转换工具
import torch.nn.functional as F  # 导入函数式接口（如损失函数）

# 导入 Flow Matching 模型
from fm_aspp import FlowMatchingModel  # 从 df.py 中导入自定义的 FlowMatchingModel 类

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体以防中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# ==================== 数据集类 ====================  # 分隔注释：数据集定义部分
class RadioMapDataset(Dataset):  # 定义自定义数据集类，继承 PyTorch Dataset
    """
    无线信道地图数据集  # 数据集描述
    输入：城市地图 + 发射机位置 (2通道)  # 条件输入为两通道图像
    输出：完整的信道增益图 (1通道)  # 目标输出为单通道增益图
    """
    def __init__(self, data_dir="fm/RadioMapSeer", img_size=256):  # 初始化函数，设置数据路径和图像大小
        self.data_dir = data_dir  # 保存数据目录路径
        self.img_size = img_size  # 保存图像尺寸（当前未缩放，保留参数）
        self.samples = []  # 初始化样本列表（保存 map_id 与 tx_id 对）
        
        # 检查路径是否存在
        if not os.path.exists(data_dir):  # 判断数据目录是否存在
            print(f"⚠️ 警告：数据目录不存在 {data_dir}")  # 输出警告信息
            
        gain_dir = os.path.join(data_dir, "gain/DPM")  # 构建增益图路径
        if os.path.exists(gain_dir):  # 如果增益图路径存在
            for gain_file in glob.glob(os.path.join(gain_dir, "*.png")):  # 遍历所有 PNG 增益图文件
                filename = os.path.basename(gain_file)  # 获取文件名
                parts = filename.replace('.png', '').split('_')  # 解析文件名中的 map_id 和 tx_id
                if len(parts) == 2:  # 如果格式正确（两个部分）
                    map_id, tx_id = int(parts[0]), int(parts[1])  # 转为整数
                    self.samples.append((map_id, tx_id))  # 添加样本对到列表
        
        print(f"✅ 找到 {len(self.samples)} 个数据样本")  # 输出样本数量
        
        self.transform = transforms.ToTensor()  # 定义图像到张量的转换
        self.thresh = 0.2  # 噪声基底阈值（用于归一化增益图）
    
    def __len__(self):  # 返回数据集长度
        return len(self.samples)  # 样本数量即为数据集长度
    
    def __getitem__(self, idx):  # 根据索引获取一个样本
        map_id, tx_id = self.samples[idx]  # 解析出地图编号与发射机编号
        
        # 1. 加载城市地图
        building_path = os.path.join(self.data_dir, "png/buildings_complete", f"{map_id}.png")  # 拼接建筑物图像路径
        building_img = np.array(Image.open(building_path).convert('L')) / 255.0  # 读取为灰度并归一化到 [0,1]
        building_tensor = self.transform(building_img).type(torch.float32)  # 转为张量并设为 float32
        
        # 2. 加载发射机位置
        tx_path = os.path.join(self.data_dir, "png/antennas", f"{map_id}_{tx_id}.png")  # 拼接发射机图像路径
        tx_img = np.array(Image.open(tx_path).convert('L')) / 255.0  # 读取为灰度并归一化到 [0,1]
        tx_tensor = self.transform(tx_img).type(torch.float32)  # 转为张量并设为 float32
        building_tensor = building_tensor * 2.0 - 1.0
        tx_tensor = tx_tensor * 2.0 - 1.0
        condition = torch.cat([building_tensor, tx_tensor], dim=0)
        # 3. 加载增益图 (Ground Truth)
        gain_path = os.path.join(self.data_dir, "gain/DPM", f"{map_id}_{tx_id}.png")  # 拼接增益图路径
        gain_img = np.expand_dims(np.array(Image.open(gain_path).convert('L')), axis=2) / 255.0  # 读取为灰度并扩展通道维
        
        # === 关键修改：归一化到 [-1, 1] ===
        # 原始数据在 [thresh, 1.0] 之间，先映射到 [0, 1]，再映射到 [-1, 1]
        mask = gain_img < self.thresh  # 找出低于阈值的像素位置
        gain_img[mask] = self.thresh  # 将低值提升到阈值，避免过低噪声影响
        gain_img = (gain_img - self.thresh) / (1 - self.thresh)  # 将 [thresh,1] 映射到 [0,1]
        gain_img = gain_img * 2.0 - 1.0  # 将 [0,1] 映射到 [-1,1]
        
        gain_tensor = self.transform(gain_img).type(torch.float32)  # 增益图转为张量
        
        condition = torch.cat([building_tensor, tx_tensor], dim=0)  # 条件张量：建筑物与发射机通道拼接
        
        return {  # 返回一个字典作为批次项
            'condition': condition,  # 条件张量 [2,H,W]
            'gain_map': gain_tensor,  # 增益图张量 [1,H,W]
            'map_id': map_id,  # 地图编号
            'tx_id': tx_id  # 发射机编号
        }


if __name__ == "__main__":  # 判断是否为主程序入口
    main()  # 调用主函数启动程序
