import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 函数式API
import math  # 数学库，用于计算对数等数学运算

# ============================================================
# 1. 基础组件 (Basic Components)
# ============================================================

class TimeEmbedding(nn.Module):
    """
    时间步嵌入模块
    作用：将连续的时间 t [0, 1] 编码为高维向量
    使用正弦-余弦位置编码（类似于 Transformer 中的位置编码）
    """
    def __init__(self, dim):
        # 初始化时间嵌入模块
        super(TimeEmbedding, self).__init__()  # 调用父类初始化函数
        self.dim = dim  # 嵌入向量的维度
        
    def forward(self, t):
        # 前向传播：将时间 t 编码为高维向量
        # t: [Batch] 范围 [0, 1]，表示当前时间步
        device = t.device  # 获取张量所在的设备（CPU或CUDA）
        half_dim = self.dim // 2  # 计算嵌入维度的一半（用于正弦和余弦）
        emb = math.log(10000) / (half_dim - 1)  # 计算频率的衰减因子（基于 Transformer 位置编码）
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 计算各个频率分量 [half_dim]
        emb = t[:, None] * emb[None, :]  # 广播相乘：时间 t 乘以频率 [Batch, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # 拼接正弦和余弦编码 [Batch, dim]
        return emb  # 返回时间嵌入向量

class ResBlock(nn.Module):
    """
    残差块 (Residual Block)
    用于构建深层网络，通过跳跃连接缓解梯度消失问题
    包含时间嵌入的注入机制
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        # 初始化残差块
        super().__init__()  # 调用父类初始化函数
        
        # 第一个卷积分支：归一化 -> 激活 -> 卷积
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),  # 组归一化，将通道分为8组进行归一化
            nn.SiLU(),  # SiLU激活函数（Sigmoid Linear Unit，也称为 Swish）
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 3x3卷积，保持空间尺寸
        )
        
        # 时间嵌入映射层：将时间嵌入投影到输出通道数
        self.time_mlp = nn.Sequential(
            nn.SiLU(),  # SiLU激活函数
            nn.Linear(time_emb_dim, out_channels)  # 全连接层，将时间嵌入映射到输出通道维度
        )
        
        # 第二个卷积分支：归一化 -> 激活 -> Dropout -> 卷积
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),  # 组归一化
            nn.SiLU(),  # SiLU激活函数
            nn.Dropout(dropout),  # Dropout正则化，防止过拟合
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 3x3卷积
        )
        
        # 残差连接的快捷路径：如果输入输出通道数不同，需要1x1卷积调整通道数
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积调整通道数
        else:
            self.shortcut = nn.Identity()  # 通道数相同时，使用恒等映射

    def forward(self, x, t_emb):
        # 前向传播
        # x: 输入特征图 [B, in_channels, H, W]
        # t_emb: 时间嵌入 [B, time_emb_dim]
        h = self.conv1(x)  # 第一次卷积 [B, out_channels, H, W]
        time_emb_proj = self.time_mlp(t_emb)[:, :, None, None]  # 时间嵌入投影并扩展空间维度 [B, out_channels, 1, 1]
        h = h + time_emb_proj  # 将时间信息加到特征图上（广播加法）
        h = self.conv2(h)  # 第二次卷积 [B, out_channels, H, W]
        return h + self.shortcut(x)  # 残差连接：输出 = 主路径 + 快捷路径


# ============================================================
# 2. 新增增强模块：ASPP 和 CBAM
# ============================================================

class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module(卷积块注意力模块)
    结合了通道注意力(Channel Attention)和空间注意力(Spatial Attention)
    用于自适应地重新校准通道和空间特征的重要性
    """
    def __init__(self, channels, reduction=16):
        # 初始化CBAM模块
        super(CBAM, self).__init__()  # 调用父类初始化函数
        # 通道注意力门控网络
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应全局平均池化，输出 [B, C, 1, 1]
            nn.Conv2d(channels, channels // reduction, 1, bias=False),  # 1x1卷积降维，减少参数量
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(channels // reduction, channels, 1, bias=False),  # 1x1卷积恢复维度
            nn.Sigmoid()  # Sigmoid激活，输出0-1之间的注意力权重
        )
        # 空间注意力门控网络
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),  # 7x7卷积，输入2通道(max+avg)，输出1通道
            nn.Sigmoid()  # Sigmoid激活，输出0-1之间的空间注意力权重
        )

    def forward(self, x):
        # 前向传播
        # x: 输入特征图 [B, C, H, W]
        # 1. Channel Attention(通道注意力)
        # 这里的实现略作简化，使用了AvgPool产生的权重
        x_out = self.channel_gate(x) * x  # 通道注意力加权：元素级相乘 [B, C, H, W]
        
        # 2. Spatial Attention(空间注意力)
        # 在通道维度上做 MaxPool 和 AvgPool，然后拼接
        max_pool = torch.max(x_out, dim=1, keepdim=True)[0]  # 沿通道维度取最大值 [B, 1, H, W]
        avg_pool = torch.mean(x_out, dim=1, keepdim=True)  # 沿通道维度取平均值 [B, 1, H, W]
        spatial_out = self.spatial_gate(torch.cat([max_pool, avg_pool], dim=1))  # 拼接后通过卷积生成空间注意力图 [B, 1, H, W]
        
        return x_out * spatial_out  # 空间注意力加权：元素级相乘 [B, C, H, W]

class ASPP(nn.Module):
    """
    ASPP: Atrous Spatial Pyramid Pooling(空洞空间金字塔池化)
    通过多尺度空洞卷积扩大感受野，捕捉多尺度上下文信息
    来源于 DeepLab 系列语义分割网络
    """
    def __init__(self, in_channels, out_channels):
        # 初始化ASPP模块
        super(ASPP, self).__init__()  # 调用父类初始化函数
        
        # 分支1: 全局平均池化 (Global Context) - 捕获全局上下文信息
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应全局平均池化，输出 [B, C, 1, 1]
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1卷积调整通道数
            nn.GroupNorm(8, out_channels),  # 组归一化
            nn.SiLU()  # SiLU激活函数
        )
        
        # 分支2: 1x1 卷积 - 捕获点级特征
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # 分支3: 3x3 空洞卷积, dilation rate=6 - 小尺度感受野
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        # 分支4: 3x3 空洞卷积, dilation rate=12 - 中等尺度感受野
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        # 分支5: 3x3 空洞卷积, dilation rate=18 - 大尺度感受野
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)

        # 融合层：将5个分支的特征融合
        self.bn = nn.GroupNorm(32, out_channels * 5)  # 组归一化(5个分支拼接后的通道数)
        self.relu = nn.SiLU()  # SiLU激活函数
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)  # 1x1卷积融合特征
        self.dropout = nn.Dropout(0.1)  # Dropout正则化，防止过拟合

    def forward(self, x):
        # 前向传播
        # x: 输入特征图 [B, in_channels, H, W]
        h = x.shape[2]  # 获取特征图的高度
        w = x.shape[3]  # 获取特征图的宽度
        
        # 1. 全局特征并上采样回原尺寸
        x_avg = self.global_avg_pool(x)  # 全局池化 [B, out_channels, 1, 1]
        x_avg = F.interpolate(x_avg, size=(h, w), mode='bilinear', align_corners=True)  # 双线性插值上采样 [B, out_channels, H, W]
        
        # 2. 多尺度空洞卷积 - 并行提取不同尺度的特征
        x1 = self.conv1(x)  # 1x1卷积分支 [B, out_channels, H, W]
        x2 = self.conv2(x)  # 空洞率=6的分支 [B, out_channels, H, W]
        x3 = self.conv3(x)  # 空洞率=12的分支 [B, out_channels, H, W]
        x4 = self.conv4(x)  # 空洞率=18的分支 [B, out_channels, H, W]
        
        # 3. 拼接并融合 - 将5个分支的特征在通道维度拼接
        x_cat = torch.cat([x_avg, x1, x2, x3, x4], dim=1)  # 拼接 [B, out_channels*5, H, W]
        x_cat = self.bn(x_cat)  # 组归一化
        x_cat = self.relu(x_cat)  # 激活
        x_cat = self.dropout(x_cat)  # Dropout
        
        return self.final_conv(x_cat)  # 1x1卷积融合 [B, out_channels, H, W]

# ============================================================
# 3. 编码器与解码器 (Encoder & Decoder)
# ============================================================

class Encoder(nn.Module):
    """
    编码器：将输入特征图逐层压缩，提取多尺度特征
    采用U-Net架构，保存中间特征用于跳跃连接
    """
    def __init__(self, input_dim, hidden_dim, num_layers, time_emb_dim):
        # 初始化编码器
        super().__init__()  # 调用父类初始化函数
        self.input_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)  # 输入卷积层，将输入通道映射到hidden_dim
        self.res_blocks = nn.ModuleList()  # 残差块列表
        self.downsamples = nn.ModuleList()  # 下采样层列表
        
        curr_dim = hidden_dim  # 当前层的通道数
        for i in range(num_layers):  # 遍历每一层
            # 通道数逐层翻倍: 64 -> 128 -> 256 -> 512 (最多512)
            out_dim = hidden_dim * (2 ** min(i, 3))  # 计算当前层输出通道数，最多翻8倍(2^3)
            self.res_blocks.append(ResBlock(curr_dim, out_dim, time_emb_dim))  # 添加残差块
            if i < num_layers - 1:  # 如果不是最后一层
                self.downsamples.append(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)  # 步长为2的卷积进行下采样，空间尺寸减半
                )
            else:  # 最后一层不下采样
                self.downsamples.append(nn.Identity())  # 恒等映射
            curr_dim = out_dim  # 更新当前通道数
            
    def forward(self, x, t_emb):
        # 前向传播
        # x: 输入特征图 [B, input_dim, H, W]
        # t_emb: 时间嵌入 [B, time_emb_dim]
        skips = []  # 用于存储跳跃连接的特征
        x = self.input_conv(x)  # 输入卷积 [B, hidden_dim, H, W]
        for block, downsample in zip(self.res_blocks, self.downsamples):  # 遍历每层
            x = block(x, t_emb)  # 通过残差块 [B, out_dim, H, W]
            skips.append(x)  # 保存当前特征用于跳跃连接
            x = downsample(x)  # 下采样 [B, out_dim, H/2, W/2]
        return x, skips  # 返回最终特征和所有跳跃连接特征

class Decoder(nn.Module):
    """
    解码器：将压缩的特征逐层恢复，结合跳跃连接重建空间细节
    采用U-Net架构的解码器设计
    """
    def __init__(self, hidden_dim, num_layers, time_emb_dim):
        # 初始化解码器
        super().__init__()  # 调用父类初始化函数
        channels = [hidden_dim * (2 ** min(i, 3)) for i in range(num_layers)]  # 计算每层的通道数列表
        channels_reversed = channels[::-1]  # 反转通道数列表(从深层到浅层)
        
        self.res_blocks = nn.ModuleList()  # 残差块列表
        self.upsamples = nn.ModuleList()  # 上采样层列表
        
        for i in range(num_layers):  # 遍历每一层
            curr_ch = channels_reversed[i]  # 当前层通道数
            out_ch = channels_reversed[i+1] if i < num_layers - 1 else hidden_dim  # 输出通道数
            
            skip_ch = curr_ch  # 跳跃连接的通道数(来自编码器对应层)
            res_in_ch = curr_ch + skip_ch  # 残差块输入通道数 = 当前特征 + 跳跃连接特征
            
            self.res_blocks.append(ResBlock(res_in_ch, out_ch, time_emb_dim))  # 添加残差块
            
            if i < num_layers - 1:  # 如果不是最后一层
                self.upsamples.append(
                    nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)  # 转置卷积上采样，空间尺寸翻倍
                )
            else:  # 最后一层不上采样
                self.upsamples.append(nn.Identity())  # 恒等映射

        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 最终1x1卷积，输出单通道预测

    def forward(self, x, skips, t_emb):
        # 前向传播
        # x: 编码器输出的瓶颈特征 [B, C, H', W']
        # skips: 编码器的跳跃连接特征列表
        # t_emb: 时间嵌入 [B, time_emb_dim]
        skips = skips[::-1]  # 反转跳跃连接列表(从深层到浅层)
        for i, (upsample, block) in enumerate(zip(self.upsamples, self.res_blocks)):  # 遍历每层
            if i < len(skips):  # 如果有跳跃连接
                skip = skips[i]  # 获取对应的跳跃连接特征
                x = torch.cat([x, skip], dim=1)  # 在通道维度拼接当前特征和跳跃连接 [B, curr_ch+skip_ch, H, W]
            x = block(x, t_emb)  # 通过残差块 [B, out_ch, H, W]
            x = upsample(x)  # 上采样 [B, out_ch, H*2, W*2]
        return self.final_conv(x)  # 最终卷积输出单通道结果 [B, 1, H, W]


# ============================================================
# 4. U-Net 主体 (集成 ASPP + Attention)
# ============================================================

class UNet(nn.Module):
    """
    U-Net主干网络，增强版本集成了ASPP和CBAM模块
    用于预测Flow Matching中的速度场
    """
    def __init__(self, input_dim, hidden_dim, num_layers, time_emb_dim=256):
        # 初始化U-Net
        super().__init__()  # 调用父类初始化函数
        self.time_embedding = TimeEmbedding(hidden_dim)  # 时间嵌入模块
        self.time_mlp = nn.Sequential(  # 时间嵌入的多层感知机，将时间嵌入映射到更高维度
            nn.Linear(hidden_dim , time_emb_dim),  # 线性层扩展维度
            nn.SiLU(),  # SiLU激活
            nn.Linear(time_emb_dim, time_emb_dim),  # 再次线性变换
        )
        
        # 编码器
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, time_emb_dim)
        
        # === 瓶颈层增强模块 ===
        # 计算编码器最后一层的输出通道数
        # 对应 Encoder 中的逻辑: hidden_dim * (2 ** min(num_layers-1, 3))
        # 默认 num_layers=4, hidden_dim=64 -> bottleneck_dim = 64 * 8 = 512
        bottleneck_dim = hidden_dim * (2 ** min(num_layers-1, 3))  # 计算瓶颈层通道数
        
        # 1. ASPP 模块 (扩大感受野，捕获多尺度上下文)
        self.aspp = ASPP(bottleneck_dim, bottleneck_dim)
        
        # 2. CBAM 模块 (增强关键特征，通道和空间注意力)
        self.attention = CBAM(bottleneck_dim)
        # ====================
        
        # 解码器
        self.decoder = Decoder(hidden_dim, num_layers, time_emb_dim)

    def forward(self, x, t):
        # 前向传播
        # x: 输入特征 [B, input_dim, H, W]
        # t: 时间步 [B] 或标量
        if t.dim() == 0:  # 如果t是标量
            t = t.unsqueeze(0).repeat(x.shape[0])  # 扩展为batch维度 [B]
            
        t_emb = self.time_embedding(t)  # 时间嵌入 [B, hidden_dim]
        t_emb = self.time_mlp(t_emb)  # 映射到更高维度 [B, time_emb_dim]
        
        # Encoder: 编码过程，提取多尺度特征
        x, skips = self.encoder(x, t_emb)  # x: [B, bottleneck_dim, H', W'], skips: 跳跃连接列表
        
        # === 瓶颈层处理 ===
        # 依次通过 ASPP 和 Attention，增强瓶颈层特征
        x = self.aspp(x)  # ASPP多尺度特征提取 [B, bottleneck_dim, H', W']
        x = self.attention(x)  # CBAM注意力增强 [B, bottleneck_dim, H', W']
        # =================
        
        # Decoder: 解码过程，恢复空间分辨率
        x = self.decoder(x, skips, t_emb)  # 输出 [B, 1, H, W]
        return x  # 返回预测的速度场

# ============================================================
# 5. Flow Matching 模型封装
# ============================================================

class FlowMatchingModel(nn.Module):
    """
    基于 Optimal Transport Conditional Flow Matching (OT-CFM) 的模型
    该模型预测向量场 (Vector Field) v_t(x)，指导样本从噪声流向数据
    核心思想：学习从先验分布(高斯噪声)到数据分布的最优传输路径
    """
    def __init__(self, condition_dim=2, hidden_dim=64, num_layers=4):
        # 初始化Flow Matching模型
        super().__init__()  # 调用父类初始化函数
        
        # 输入 = 当前状态 x_t (1通道) + 条件 (condition_dim通道)
        input_dim = 1 + condition_dim  # 总输入通道数
        
        # 使用增强版 UNet 作为向量场预测网络 v_theta(x_t, t, condition)
        self.unet = UNet(input_dim, hidden_dim, num_layers)
        
        # 极小值，防止 t=0 或 t=1 时的数值不稳定
        self.sigma_min = 1e-4  # 数值稳定性参数

    def forward(self, x, t, condition):
        """
        预测速度场 v_t
        这是Flow Matching的核心：在时间t预测样本x应该沿着哪个方向流动
        Args:
            x: 当前状态 x_t [B, 1, H, W] - 时间t时刻的样本状态
            t: 时间步 [B] (范围 0~1) - 从噪声(t=0)到数据(t=1)的时间
            condition: 条件 [B, C, H, W] - 控制生成内容的条件信息(如建筑图、发射机位置)
        Returns:
            v_pred: 预测的速度场 [B, 1, H, W] - 指示样本流动方向
        """
        # 拼接输入：将当前状态和条件信息在通道维度拼接
        x_input = torch.cat([x, condition], dim=1)  # [B, 1+condition_dim, H, W]
        # 预测速度场：通过UNet网络预测当前时刻的流动速度
        v_pred = self.unet(x_input, t)  # [B, 1, H, W]
        return v_pred  # 返回预测的速度场

    @torch.no_grad()  # 禁用梯度计算，加速推理
    def sample(self, condition, steps=5, device='cuda', solver='euler'):   # 在这里进行修改 步数
        """
        使用 ODE Solver 进行采样 (生成)
        从随机噪声出发，沿着学习到的向量场流动，最终生成目标数据
        Args:
            condition: 条件信息 [B, C, H, W]
            steps: ODE求解步数，步数越多结果越精确但速度越慢
            device: 计算设备
            solver: 求解器类型，可选 'euler' 或 'heun'
        Returns:
            生成的样本 [B, 1, H, W]
        """
        # 记录进入时的模式，避免推理后把外部训练状态打乱
        was_training = self.training
        self.eval()  # 设置为评估模式
        B, _, H, W = condition.shape  # 获取batch大小和空间尺寸
        
        # 1. 初始化噪声 x_0 ~ N(0, 1) - 从标准高斯分布采样初始状态
        x = torch.randn(B, 1, H, W, device=device)  # [B, 1, H, W]
        
        # 2. 定义时间网格 (从 0 到 1) - 将时间区间[0,1]均匀分割
        dt = 1.0 / steps  # 时间步长
        time_steps = torch.linspace(0, 1, steps + 1, device=device)  # 时间点序列 [0, dt, 2dt, ..., 1]
        
        # 3. 数值积分求解 ODE - 常微分方程 dx/dt = v(x,t)
        for i in range(steps):  # 遍历每个时间步
            t_curr = time_steps[i]  # 当前时间点
            
            # 扩展 t 以匹配 batch size
            t_batch = torch.full((B,), t_curr.item(), device=device, dtype=torch.float32)  # [B]
            
            # 预测速度场：在当前时刻和状态下预测流动方向
            v_pred = self.forward(x, t_batch, condition)  # [B, 1, H, W]

            if solver == 'heun':
                # Heun (RK2) 修正：先做一次 Euler 预测，再用 t+dt 的速度场校正
                t_next = time_steps[i + 1]
                t_batch_next = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
                x_euler = x + v_pred * dt
                v_pred_next = self.forward(x_euler, t_batch_next, condition)
                x = x + 0.5 * dt * (v_pred + v_pred_next)
            else:
                # 默认 Euler 更新：x(t+dt) = x(t) + v(x,t) * dt
                x = x + v_pred * dt  # 沿速度场方向前进一小步
            
        if was_training:
            self.train()  # 仅在进入时为训练模式才恢复
        
        # 截断数值范围到[-1,1]，确保输出在有效范围内
        return torch.clamp(x, -1, 1)  # [B, 1, H, W]

if __name__ == '__main__':
    # ============================================================
    # 测试代码：验证模型结构是否正确
    # ============================================================
    print("正在测试 Enhanced Flow Matching Model 结构...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选择可用设备
    
    # 测试实例化
    model = FlowMatchingModel(condition_dim=2, hidden_dim=32, num_layers=3).to(device)  # 创建模型并移到设备
    
    # 模拟输入
    B, H, W = 2, 64, 64  # Batch大小=2，图像尺寸=64x64
    x_t = torch.randn(B, 1, H, W).to(device)  # 随机初始化当前状态
    condition = torch.randn(B, 2, H, W).to(device)  # 随机初始化条件(2通道)
    t = torch.rand(B).to(device)  # 随机时间步 [0,1]
    
    # 测试 Forward（前向传播）
    v_pred = model(x_t, t, condition)  # 前向传播
    print(f"输入尺寸: {x_t.shape}")  # 打印输入尺寸
    print(f"输出速度场尺寸: {v_pred.shape}")  # 打印输出尺寸
    
    # 验证输出尺寸是否正确
    if v_pred.shape == x_t.shape:  # 输出尺寸应该与输入相同
        print("✅ 测试通过：输出尺寸正确！")
        # 打印一下参数量，确认新模块已加入
        num_params = sum(p.numel() for p in model.parameters())  # 计算总参数量
        print(f"模型参数量: {num_params:,}")  # 打印参数量（千分位分隔）
    else:
        print("❌ 测试失败：尺寸不匹配")  # 测试失败
