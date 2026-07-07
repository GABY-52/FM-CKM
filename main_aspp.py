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

# ==================== 训练函数 (Flow Matching) ====================  # 分隔注释：训练函数定义
def train_fm_model(  # 定义流匹配训练函数
    model,  # 模型实例
    dataloader,  # 训练数据加载器
    num_epochs=100,  # 训练轮数（默认 100）
    learning_rate=1e-4,  # 学习率（默认 1e-4）
    device='cuda',  # 训练设备（默认 CUDA）
    save_dir='fm/ASPP/saved_models',  # 模型保存目录
    start_epoch=0  # 起始轮数（支持断点续训）
):
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录（若不存在）
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # 使用 AdamW 优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  # 余弦退火学习率调度
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数（拟合速度场）
    train_losses = []  # 记录每轮平均损失
    
    print(f"🚀 开始训练 Flow Matching 模型 (OT-CFM)...")  # 输出训练开始信息
    
    for epoch in range(start_epoch, num_epochs):  # 遍历每一训练轮次
        model.train()  # 切换到训练模式
        epoch_loss = 0.0  # 初始化当轮损失累积
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")  # 创建进度条
        
        for batch in progress_bar:  # 遍历每一个批次
            # 数据准备
            condition = batch['condition'].to(device)  # 条件张量移到设备
            x_1 = batch['gain_map'].to(device)  # 真实增益图（数据端点）
            batch_size = x_1.shape[0]  # 获取批次大小
            
            # === Optimal Transport Conditional Flow Matching (OT-CFM) 核心逻辑 ===
            
            # 1. 采样 x_0 (标准高斯噪声)
            x_0 = torch.randn_like(x_1).to(device)  # 与 x_1 尺寸一致的高斯噪声
            
            # 2. 采样时间 t ~ Uniform[0, 1]
            #t = torch.rand(batch_size, device=device)  # 在 [0,1] 均匀采样时间
            # 2) t 采样（下一节我还建议换成 Beta）
            t = torch.rand(batch_size, device=device)
            t_expand = t.view(batch_size, 1, 1, 1)
            x_0 = torch.randn_like(x_1)
            # --- Gaussian-smoothed bridge ---
            eps = torch.randn_like(x_1)

            # 3. 构造中间状态 x_t (线性插值路径)
            # x_t = (1 - (1-sigma_min)t) * x_0 + t * x_1  # 理论路径公式
            # 简化版 (sigma_min approx 0): x_t = (1-t)x_0 + t*x_1  # 常见近似写法
            #t_expand = t.view(batch_size, 1, 1, 1)  # 扩展时间维度用于广播
            #sigma_min = 1e-4  # 极小噪声项，提升数值稳定性

            sigma0 = 0.05  # 需要调参：0.01~0.1 都可试
            sigma_t = sigma0 * t_expand * (1 - t_expand)          # σ(t)
            sigma_prime = sigma0 * (1 - 2 * t_expand)             # σ'(t)

            x_t = (1 - t_expand) * x_0 + t_expand * x_1 + sigma_t * eps
            v_target = (x_1 - x_0) + sigma_prime * eps            # 对应论文(9)
            
            # OT路径插值公式
            #mu_t = t_expand * x_1 + (1 - (1 - sigma_min) * t_expand) * x_0  # 定义插值路径点
            #x_t = mu_t  # 这里是确定性插值，OT-CFM 通常无需加噪声或仅加极小噪声
            
            # 4. 计算目标速度向量 (Target Velocity)
            # u_t(x|x_1) = (x_1 - (1-sigma_min)x_0) / (1 - (1-sigma_min)t) 的流场导数  # 说明参考公式
            # 对于 OT 路径，v_target 就是简单的直线斜率: x_1 - (1-sigma_min)x_0  # 简化后的目标
            #v_target = x_1 - (1 - sigma_min) * x_0  # 目标速度场（与 t 无关）
            
            # 5. 模型预测
            #v_pred = model(x_t, t, condition)  # 预测速度场 v_theta(x_t, t, condition)
            
            v_pred = model(x_t, t, condition)
            loss = criterion(v_pred, v_target)
            # 6. 计算损失 (MSE)
            #loss = criterion(v_pred, v_target)  # 使用 MSE 衡量预测速度与目标速度的差异
            
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()  # 更新模型参数
            
            epoch_loss += loss.item()  # 累加当前批次损失
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})  # 在进度条上显示当前损失
        
        avg_loss = epoch_loss / len(dataloader)  # 计算当轮平均损失
        train_losses.append(avg_loss)  # 记录平均损失
        scheduler.step()  # 更新学习率（余弦退火）
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")  # 打印当轮训练信息
        
        # 保存模型
        if (epoch + 1) % 10 == 0:  # 每 10 轮保存一次模型检查点
            checkpoint = {  # 构建检查点字典
                'epoch': epoch,  # 当前轮次
                'model_state_dict': model.state_dict(),  # 模型权重
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
                'loss': avg_loss,  # 当轮平均损失
            }
            save_path = os.path.join(save_dir, f'fm_epoch_{epoch+1}.pth')  # 保存路径
            torch.save(checkpoint, save_path)  # 保存到磁盘
            print(f"💾 模型已保存: {save_path}")  # 打印保存提示
        
        # 可视化
        if (epoch + 1) % 10 == 0:  # 每 20 轮进行一次生成可视化
            visualize_generation(model, dataloader, device, epoch+1, save_dir)  # 生成并保存可视化结果
            
    return train_losses  # 返回训练损失列表

# ==================== 可视化函数 ====================  # 分隔注释：可视化函数定义
def visualize_generation(model, dataloader, device, epoch, save_dir):  # 定义可视化生成函数
    model.eval()  # 切换到评估模式
    batch = next(iter(dataloader))  # 获取一个批次数据
    condition = batch['condition'][:4].to(device)  # 取前 4 个样本的条件张量
    real_gain = batch['gain_map'][:4].to(device)  # 取前 4 个样本的真实增益图
    
    # 使用 ODE Solver 生成 (20步)
    with torch.no_grad():  # 禁用梯度计算，加速推理
        # 生成结果在 [-1, 1]
        generated_gain = model.sample(condition, steps=20, device=device)  # 采样生成增益图  在这里修改 步数
        
    # 反归一化到 [0, 1] 用于显示
    generated_gain = (generated_gain + 1) / 2  # 将生成结果从 [-1,1] 映射到 [0,1]
    real_gain = (real_gain + 1) / 2  # 将真实增益图从 [-1,1] 映射到 [0,1]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))  # 创建 4x4 子图用于展示
    for i in range(4):  # 遍历 4 个样本
        axes[i, 0].imshow(condition[i, 0].cpu().numpy(), cmap='gray')  # 显示建筑物通道
        axes[i, 0].set_title('Building')  # 设置标题
        axes[i, 0].axis('off')  # 关闭坐标轴
        
        axes[i, 1].imshow(condition[i, 1].cpu().numpy(), cmap='hot')  # 显示发射机通道
        axes[i, 1].set_title('Transmitter')  # 设置标题
        axes[i, 1].axis('off')  # 关闭坐标轴
        
        axes[i, 2].imshow(generated_gain[i, 0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)  # 显示生成结果
        axes[i, 2].set_title('FM Generated')  # 设置标题
        axes[i, 2].axis('off')  # 关闭坐标轴
        
        axes[i, 3].imshow(real_gain[i, 0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)  # 显示真实增益图
        axes[i, 3].set_title('Ground Truth')  # 设置标题
        axes[i, 3].axis('off')  # 关闭坐标轴
    
    plt.tight_layout()  # 自动调整布局以避免重叠
    plt.savefig(os.path.join(save_dir, f'vis_epoch_{epoch}.png'))  # 保存可视化结果到文件
    plt.close()  # 关闭图像以释放内存

# ==================== 评估函数 ====================  # 分隔注释：评估函数定义
def evaluate_model(model, dataloader, device, max_samples=100, steps=5):  # 定义评估函数
    model.eval()  # 切换到评估模式
    mse_scores = []  # 初始化 MSE 分数列表
    
    print(f"\n🔍 评估模型 (ODE Steps={steps})...")  # 输出评估信息
    with torch.no_grad():  # 禁用梯度，提升评估速度
        count = 0  # 已评估样本计数
        for batch in tqdm(dataloader):  # 遍历评估数据集
            if count >= max_samples: break  # 达到最大评估样本数则停止
            
            condition = batch['condition'].to(device)  # 条件张量移到设备
            real_gain = batch['gain_map'].to(device) # [-1, 1]  # 真实增益图（范围 [-1,1]）
            
            # 生成
            generated = model.sample(condition, steps=steps, device=device) # [-1, 1]  # 生成增益图（范围 [-1,1]）
            
            # 反归一化回 [0, 1] 进行物理意义上的 MSE 计算
            gen_phys = (generated + 1) / 2  # 生成图转为 [0,1]
            real_phys = (real_gain + 1) / 2  # 真实图转为 [0,1]
            
            mse = F.mse_loss(gen_phys, real_phys, reduction='none').view(condition.size(0), -1).mean(1)  # 逐样本计算 MSE
            mse_scores.extend(mse.cpu().tolist())  # 添加到列表
            count += condition.size(0)  # 增加计数
            
    avg_mse = np.mean(mse_scores)  # 计算平均 MSE
    print(f"📊 平均 MSE: {avg_mse:.6f}")  # 打印评估结果
    return avg_mse  # 返回平均 MSE

# ==================== 主函数 ====================  # 分隔注释：主函数入口
def main():  # 主程序入口函数
    BATCH_SIZE = 8  # 训练批次大小
    NUM_EPOCHS = 100  # 训练轮数
    LEARNING_RATE = 1e-4  # 学习率
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择运行设备
    
    # ===== 固定随机种子以确保训练/测试集划分可复现 =====
    import random  # 导入 Python 标准随机库
    SEED = 42  # 设置随机种子为 42（可以是任意整数）
    random.seed(SEED)  # 固定 Python 随机数生成器
    torch.manual_seed(SEED)  # 固定 PyTorch CPU 随机数生成器
    np.random.seed(SEED)  # 固定 NumPy 随机数生成器
    if torch.cuda.is_available():  # 如果使用 GPU
        torch.cuda.manual_seed(SEED)  # 固定 PyTorch GPU 随机数生成器
        torch.cuda.manual_seed_all(SEED)  # 固定所有 GPU 的随机数生成器
    print(f"🎲 随机种子已固定: {SEED}")  # 输出随机种子信息
    # ========================================================
    
    print(f"🖥️ 设备: {DEVICE}")  # 输出当前使用设备
    
    dataset = RadioMapDataset(data_dir="fm/RadioMapSeer")  # 初始化数据集
    if len(dataset) == 0:  # 如果数据集为空
        print("❌ 未找到数据，请检查路径。")  # 输出错误提示
        return  # 退出程序

    train_size = int(0.8 * len(dataset))  # 划分训练集大小（80%）
    test_size = len(dataset) - train_size  # 划分测试集大小（20%）
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # 随机划分训练/测试集
    
    # ===== 保存测试集索引，用于后续推理验证 =====
    test_indices = test_dataset.indices  # 获取测试集在原数据集中的索引列表
    test_samples = [dataset.samples[i] for i in test_indices]  # 根据索引获取测试集的 (map_id, tx_id) 对
    
    os.makedirs('ASPP/save_models', exist_ok=True)  # 确保保存目录存在
    test_samples_path = 'fm/ASPP/saved_models/test_samples.json'  # 定义保存路径
    with open(test_samples_path, 'w') as f:  # 打开文件用于写入
        json.dump(test_samples, f, indent=2)  # 保存为格式化的 JSON 文件
    print(f"✅ 测试集样本已保存: {len(test_samples)} 个样本 → {test_samples_path}")  # 输出保存信息
    print(f"   训练集: {len(train_dataset)} 样本 | 测试集: {len(test_dataset)} 样本")  # 输出划分统计
    # =============================================
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # 训练数据加载器
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 测试数据加载器
    
    # 初始化 Flow Matching 模型
    model = FlowMatchingModel(  # 创建流匹配模型实例
        condition_dim=2,  # 条件通道为 2（建筑物 + 发射机）
        hidden_dim=64,  # 隐藏维度 64（模型容量）
        num_layers=4  # U-Net 层数 4（网络深度）
    ).to(DEVICE)  # 移动模型到设备
    
    # 训练
    train_fm_model(  # 调用训练函数
        model, train_loader,  # 模型与训练数据
        num_epochs=NUM_EPOCHS,  # 训练轮数
        learning_rate=LEARNING_RATE,  # 学习率
        device=DEVICE,  # 设备
        save_dir='fm/ASPP/saved_models'  # 模型保存路径
    )
    
    # 评估
    evaluate_model(model, test_loader, DEVICE, max_samples=100, steps=5)  # 在测试集上评估模型
    
    # 保存最终模型
    torch.save(model.state_dict(), 'fm/ASPP/saved_models/fm_final.pth')  # 保存最终模型权重（state_dict）
    print("✅ 全部完成")  # 输出完成提示

if __name__ == "__main__":  # 判断是否为主程序入口
    main()  # 调用主函数启动程序
