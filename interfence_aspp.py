"""
Inference Script for Flow Matching Model
Flow Matching 推理脚本 - 快速生成信道增益图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import os
import argparse
from PIL import Image
import time

from fm_aspp import FlowMatchingModel

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class InferenceEngine:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"📦 加载 Flow Matching 模型: {model_path}")
        # 注意：这里加载的是整个 checkpoint 字典还是仅 state_dict，取决于 main.py 怎么保存的
        # main.py 代码中是保存了 dict，所以需要按 key 读取
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        except Exception as e:
            # 加载失败直接抛出，避免后续出现半初始化对象
            raise RuntimeError(f"模型加载失败: {e}")

        self.model = FlowMatchingModel(
            condition_dim=2,
            hidden_dim=64, # 需与训练一致
            num_layers=4   # 需与训练一致
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✅ 模型加载成功！设备: {self.device}")
        
        self.transform = transforms.ToTensor()
        self.thresh = 0.2
    
    def load_condition_data(self, data_dir, map_id, tx_id):
        """加载数据并进行与训练时一致的预处理"""
        
        # 1. 建筑物
        building_path = os.path.join(data_dir, "png/buildings_complete", f"{map_id}.png")
        if not os.path.exists(building_path):
            raise FileNotFoundError(f"文件不存在: {building_path}")
        building = np.array(Image.open(building_path).convert('L')) / 255.0
        building_tensor = self.transform(building).float()
        
        # 2. 发射机
        tx_path = os.path.join(data_dir, "png/antennas", f"{map_id}_{tx_id}.png")
        if not os.path.exists(tx_path):
            raise FileNotFoundError(f"文件不存在: {tx_path}")
        tx = np.array(Image.open(tx_path).convert('L')) / 255.0
        tx_tensor = self.transform(tx).float()
        
        building_tensor = building_tensor * 2.0 - 1.0
        tx_tensor = tx_tensor * 2.0 - 1.0
        condition = torch.cat([building_tensor, tx_tensor], dim=0).unsqueeze(0)

        # 3. 组合条件
        #condition = torch.cat([building_tensor, tx_tensor], dim=0).unsqueeze(0) # [1, 2, H, W]
        
        # 4. 真实增益图 (用于对比)
        real_gain = None
        gain_path = os.path.join(data_dir, "gain/DPM", f"{map_id}_{tx_id}.png")
        if os.path.exists(gain_path):
            gain_img = np.array(Image.open(gain_path).convert('L')) / 255.0
            
            # 关键：完全复现训练时的归一化逻辑
            mask = gain_img < self.thresh
            gain_img[mask] = self.thresh
            gain_img = (gain_img - self.thresh) / (1 - self.thresh)  # [0, 1]
            gain_img = gain_img * 2.0 - 1.0  # [0,1] -> [-1,1] 与训练一致！
            real_gain = self.transform(gain_img).float().unsqueeze(0)
            
        return condition, building_tensor, tx_tensor, real_gain
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='fm/ASPP/saved_models/fm_final.pth')
    parser.add_argument('--data_dir', type=str, default='fm/RadioMapSeer')
    parser.add_argument('--map_id', type=int, default=594)
    parser.add_argument('--tx_id', type=int, default=12)
    parser.add_argument('--steps', type=int, default=1, help="ODE solver steps (speed vs quality)")
    parser.add_argument('--compare_steps', action='store_true', help="对比不同步数的效果（使用相同噪声）")
    parser.add_argument('--step_list', type=str, default='1,2,3,5,10,15,20,50', help="对比的步数列表，逗号分隔，如：1,5,10,20,50")
    parser.add_argument('--skip_test_check', action='store_true', help="跳过测试集检查（默认会检查）")
    
    args = parser.parse_args()
    

if __name__ == "__main__":
    main()
