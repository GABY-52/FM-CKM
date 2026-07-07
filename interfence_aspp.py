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
    
    @torch.no_grad()
    def generate(self, condition, steps=20, profile=False, solver='euler'):
        """
        生成增益图
        steps: ODE 求解步数，Flow Matching 仅需 10-20 步
        profile: 是否返回详细的性能分析
        """
        import time
        
        condition = condition.to(self.device)
        
        # 测量纯GPU计算时间
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t_compute_start = time.time()
        
        # 模型输出的是 [-1, 1] 范围，保持不变用于计算指标
        generated = self.model.sample(condition, steps=steps, device=self.device, solver=solver)
        
        # 同步 GPU 操作，确保推理完成
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t_compute_end = time.time()
        compute_time = t_compute_end - t_compute_start
        
        # 测量数据传输时间
        t_transfer_start = time.time()
        result = generated.cpu()
        t_transfer_end = time.time()
        transfer_time = t_transfer_end - t_transfer_start
        
        # 返回结果和性能信息
        if profile:
            return result, {'compute': compute_time, 'transfer': transfer_time, 'total': compute_time + transfer_time}
        return result
    
    def visualize_result(self, building, tx, generated, real=None, save_path=None):
        num_cols = 4 if real is not None else 3
        
        # 准备数据用于可视化
        gen_vis = (generated[0, 0] + 1) / 2  # [-1,1] -> [0,1]
        real_vis = None
        metrics_text = ""
        
        if real is not None:
            real_vis = (real[0, 0] + 1) / 2  # [-1,1] -> [0,1]
            
            # 计算评估指标 - 在 [-1,1] 空间计算（与训练一致）
            gen_np = generated[0, 0].numpy()
            real_np = real[0, 0].numpy()
            
            # NMSE (Normalized Mean Squared Error)
            mse = np.mean((gen_np - real_np)**2)
            # 防止全零或极小值导致除零
            nmse = mse / (np.mean(real_np**2) + 1e-8)
            
            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(mse)
            
            # SSIM (Structural Similarity Index) - 在 [0,1] 空间计算
            gen_vis_np = gen_vis.numpy()
            real_vis_np = real_vis.numpy()
            ssim_value = ssim(real_vis_np, gen_vis_np, data_range=1.0)
            
            # PSNR (Peak Signal-to-Noise Ratio) - 在 [-1,1] 空间计算，data_range=2
            psnr = 20 * np.log10(2.0 / (rmse + 1e-8)) if rmse > 0 else float('inf')
            
            metrics_text = f"NMSE: {nmse:.6f} | RMSE: {rmse:.6f} | SSIM: {ssim_value:.4f} | PSNR: {psnr:.2f} dB"
        
        # === 1. 保存合并的完整图像 ===
        fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
        
        # Map
        axes[0].imshow(building[0], cmap='gray')
        axes[0].set_title("Building Map")
        axes[0].axis('off')
        
        # Tx
        axes[1].imshow(tx[0], cmap='hot')
        axes[1].set_title("Tx Position")
        axes[1].axis('off')
        
        # Generated
        im = axes[2].imshow(gen_vis, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title("FM Generated")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        
        # Real
        if real is not None:
            im2 = axes[3].imshow(real_vis, cmap='viridis', vmin=0, vmax=1)
            axes[3].set_title("Ground Truth")
            axes[3].axis('off')
            plt.colorbar(im2, ax=axes[3], fraction=0.046)
            
            # 显示指标
            plt.figtext(0.5, 0.02, metrics_text, 
                       ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"💾 合并图像保存至: {save_path}")
        plt.show()
        
        # === 2. 分开保存并单独展示每个子图 ===
        if save_path:
            base_dir = os.path.dirname(save_path)
            base_name = os.path.splitext(os.path.basename(save_path))[0]
            
            # 2.1 Building Map
            fig1 = plt.figure(figsize=(6, 6))
            plt.imshow(building[0], cmap='gray')
            plt.title("Building Map", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            building_path = os.path.join(base_dir, f"{base_name}_building.png")
            plt.savefig(building_path, bbox_inches='tight', dpi=150)
            print(f"💾 建筑图保存至: {building_path}")
            #plt.show()
            
            # 2.2 Tx Position
            fig2 = plt.figure(figsize=(6, 6))
            plt.imshow(tx[0], cmap='hot')
            plt.title("Tx Position", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            tx_path = os.path.join(base_dir, f"{base_name}_tx.png")
            plt.savefig(tx_path, bbox_inches='tight', dpi=150)
            print(f"💾 发射机位置保存至: {tx_path}")
            #plt.show()
            
            # 2.3 FM Generated
            fig3 = plt.figure(figsize=(7, 6))
            im = plt.imshow(gen_vis, cmap='viridis', vmin=0, vmax=1)
            #plt.title("FM Generated", fontsize=14)
            #plt.colorbar(im, fraction=0.046)
            plt.axis('off')
            plt.tight_layout()
            gen_path = os.path.join(base_dir, f"{base_name}_generated.png")
            plt.savefig(gen_path, bbox_inches='tight', dpi=600, pad_inches=0)
            print(f"💾 生成结果保存至: {gen_path}")
            #plt.show()
            
            # 2.4 Ground Truth (如果存在)
            if real is not None:
                fig4 = plt.figure(figsize=(7, 6))
                im2 = plt.imshow(real_vis, cmap='viridis', vmin=0, vmax=1)
                #plt.title("Ground Truth", fontsize=14)
                #plt.colorbar(im2, fraction=0.046)
                plt.axis('off')
                plt.tight_layout()
                real_path = os.path.join(base_dir, f"{base_name}_groundtruth.png")
                plt.savefig(real_path, bbox_inches='tight', dpi=600, pad_inches=0)
                print(f"💾 真实值保存至: {real_path}")
                #plt.show()
                
                # 2.5 保存评估指标到文本文件
                metrics_file = os.path.join(base_dir, f"{base_name}_metrics.txt")
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    f.write(f"Flow Matching 评估指标\n")
                    f.write(f"=" * 50 + "\n")
                    f.write(f"NMSE: {nmse:.6f}\n")
                    f.write(f"RMSE: {rmse:.6f}\n")
                    f.write(f"SSIM: {ssim_value:.4f}\n")
                    f.write(f"PSNR: {psnr:.2f} dB\n")
                print(f"💾 评估指标保存至: {metrics_file}")
            
            print(f"\n✅ 所有图像已分别保存并展示完成！")

    def compare_different_steps(self, condition, real, step_list=[1, 5, 10, 20, 50], save_path='fm/ASPP/image/step_comparison.png'):
        """
        在同一噪声下测试不同步数的效果
        """
        print("\n" + "="*80)
        print("🔬 开始测试不同ODE步数的影响（使用相同噪声）")
        print("="*80)
        
        # 固定随机种子，确保每次使用相同的初始噪声
        torch.manual_seed(42)
        np.random.seed(42)
        
        results = {
            'steps': [],
            'ssim': [],
            'psnr': [],
            'nmse': [],
            'rmse': [],
            'time': []
        }
        
        condition = condition.to(self.device)
        
        # 预热GPU
        print("\n🔥 预热GPU...")
        _ = self.model.sample(condition, steps=5, device=self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        print(f"\n📊 测试不同步数（K = {step_list}）:")
        print("-"*80)
        
        for K in step_list:
            print(f"\n🔬 测试 K={K}...")
            
            # 重置随机种子，确保每次使用相同的初始噪声
            torch.manual_seed(42)
            
            # 测量时间（3次平均）
            times = []
            for _ in range(3):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                gen = self.model.sample(condition, steps=K, device=self.device)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1 - t0)
            
            avg_time = np.mean(times)
            gen_cpu = gen.cpu()
            
            # 计算指标
            gen_np = gen_cpu[0, 0].numpy()
            real_np = real[0, 0].numpy()
            
            # NMSE
            mse = np.mean((gen_np - real_np)**2)
            nmse = mse / np.mean(real_np**2)
            
            # RMSE & PSNR
            rmse = np.sqrt(mse)
            psnr = 20 * np.log10(2.0 / rmse) if rmse > 0 else 100.0
            
            # SSIM
            gen_vis = (gen_cpu[0, 0] + 1) / 2
            real_vis = (real[0, 0] + 1) / 2
            ssim_value = ssim(real_vis.numpy(), gen_vis.numpy(), data_range=1.0)
            
            # 保存结果
            results['steps'].append(K)
            results['ssim'].append(ssim_value)
            results['psnr'].append(psnr)
            results['nmse'].append(nmse)
            results['rmse'].append(rmse)
            results['time'].append(avg_time)
            
            print(f"   SSIM: {ssim_value:.4f}")
            print(f"   PSNR: {psnr:.2f} dB")
            print(f"   NMSE: {nmse:.6f}")
            print(f"   RMSE: {rmse:.6f}")
            print(f"   时间: {avg_time:.4f}s")
        
        # 可视化结果
        self._plot_step_comparison(results, save_path)
        
        # 保存数据到JSON
        import json
        json_path = save_path.replace('.png', '.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 数据已保存至: {json_path}")
        
        return results
    
    def _plot_step_comparison(self, results, save_path):
        """
        绘制不同步数下的指标变化折线图
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        steps = results['steps']
        
        # 1. SSIM
        ax = axes[0, 0]
        ax.plot(steps, results['ssim'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('ODE 步数 K', fontsize=12)
        ax.set_ylabel('SSIM ↑', fontsize=12)
        ax.set_title('结构相似度 (SSIM)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=max(results['ssim']), color='r', linestyle='--', alpha=0.5, 
                   label=f'最大值={max(results["ssim"]):.4f} (K={steps[np.argmax(results["ssim"])]})')
        ax.legend(loc='lower right')
        
        # 2. PSNR
        ax = axes[0, 1]
        ax.plot(steps, results['psnr'], 's-', linewidth=2, markersize=8, color='#A23B72')
        ax.set_xlabel('ODE 步数 K', fontsize=12)
        ax.set_ylabel('PSNR (dB) ↑', fontsize=12)
        ax.set_title('峰值信噪比 (PSNR)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=max(results['psnr']), color='r', linestyle='--', alpha=0.5,
                   label=f'最大值={max(results["psnr"]):.2f}dB (K={steps[np.argmax(results["psnr"])]})')
        ax.legend(loc='lower right')
        
        # 3. NMSE
        ax = axes[0, 2]
        ax.plot(steps, results['nmse'], '^-', linewidth=2, markersize=8, color='#F18F01')
        ax.set_xlabel('ODE 步数 K', fontsize=12)
        ax.set_ylabel('NMSE ↓', fontsize=12)
        ax.set_title('归一化均方误差 (NMSE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=min(results['nmse']), color='g', linestyle='--', alpha=0.5,
                   label=f'最小值={min(results["nmse"]):.6f} (K={steps[np.argmin(results["nmse"])]})')
        ax.legend(loc='upper right')
        
        # 4. RMSE
        ax = axes[1, 0]
        ax.plot(steps, results['rmse'], 'D-', linewidth=2, markersize=8, color='#6A4C93')
        ax.set_xlabel('ODE 步数 K', fontsize=12)
        ax.set_ylabel('RMSE ↓', fontsize=12)
        ax.set_title('均方根误差 (RMSE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=min(results['rmse']), color='g', linestyle='--', alpha=0.5,
                   label=f'最小值={min(results["rmse"]):.6f} (K={steps[np.argmin(results["rmse"])]})')
        ax.legend(loc='upper right')
        
        # 5. 推理时间
        ax = axes[1, 1]
        ax.plot(steps, results['time'], 'p-', linewidth=2, markersize=8, color='#C73E1D')
        ax.set_xlabel('ODE 步数 K', fontsize=12)
        ax.set_ylabel('推理时间 (s) ↓', fontsize=12)
        ax.set_title('推理时间', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加线性参考线
        if len(steps) > 1:
            linear_time = [results['time'][0] * (k / steps[0]) for k in steps]
            ax.plot(steps, linear_time, '--', alpha=0.5, color='gray', 
                   label='线性关系参考')
            ax.legend(loc='upper left')
        
        # 6. 综合对比（归一化到0-1）
        ax = axes[1, 2]
        # 归一化各指标用于对比
        ssim_norm = [(s - min(results['ssim'])) / (max(results['ssim']) - min(results['ssim']) + 1e-8) 
                     for s in results['ssim']]
        psnr_norm = [(p - min(results['psnr'])) / (max(results['psnr']) - min(results['psnr']) + 1e-8) 
                     for p in results['psnr']]
        nmse_norm = [(max(results['nmse']) - n) / (max(results['nmse']) - min(results['nmse']) + 1e-8) 
                     for n in results['nmse']]  # 反转：越小越好
        rmse_norm = [(max(results['rmse']) - r) / (max(results['rmse']) - min(results['rmse']) + 1e-8) 
                     for r in results['rmse']]  # 反转：越小越好
        
        ax.plot(steps, ssim_norm, 'o-', label='SSIM', linewidth=2, markersize=6)
        ax.plot(steps, psnr_norm, 's-', label='PSNR', linewidth=2, markersize=6)
        ax.plot(steps, nmse_norm, '^-', label='NMSE', linewidth=2, markersize=6)
        ax.plot(steps, rmse_norm, 'D-', label='RMSE', linewidth=2, markersize=6)
        ax.set_xlabel('ODE 步数 K', fontsize=12)
        ax.set_ylabel('归一化性能 (0-1)', fontsize=12)
        ax.set_title('综合性能对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 对比图表已保存至: {save_path}")
        plt.show()
        
        # 额外打印分析结果
        print("\n" + "="*80)
        print("📈 分析总结:")
        print("="*80)
        best_ssim_idx = np.argmax(results['ssim'])
        best_psnr_idx = np.argmax(results['psnr'])
        best_nmse_idx = np.argmin(results['nmse'])
        best_rmse_idx = np.argmin(results['rmse'])
        
        print(f"✓ SSIM 最优: K={steps[best_ssim_idx]} (SSIM={results['ssim'][best_ssim_idx]:.4f})")
        print(f"✓ PSNR 最优: K={steps[best_psnr_idx]} (PSNR={results['psnr'][best_psnr_idx]:.2f}dB)")
        print(f"✓ NMSE 最优: K={steps[best_nmse_idx]} (NMSE={results['nmse'][best_nmse_idx]:.6f})")
        print(f"✓ RMSE 最优: K={steps[best_rmse_idx]} (RMSE={results['rmse'][best_rmse_idx]:.6f})")
        
        # 计算性能饱和点
        ssim_range = max(results['ssim']) - min(results['ssim'])
        psnr_range = max(results['psnr']) - min(results['psnr'])
        nmse_range = max(results['nmse']) - min(results['nmse'])
        rmse_range = max(results['rmse']) - min(results['rmse'])
        
        print(f"\n✓ 指标波动范围:")
        print(f"  SSIM: {min(results['ssim']):.4f} ~ {max(results['ssim']):.4f} (范围={ssim_range:.4f})")
        print(f"  PSNR: {min(results['psnr']):.2f}dB ~ {max(results['psnr']):.2f}dB (范围={psnr_range:.2f}dB)")
        print(f"  NMSE: {min(results['nmse']):.6f} ~ {max(results['nmse']):.6f} (范围={nmse_range:.6f})")
        print(f"  RMSE: {min(results['rmse']):.6f} ~ {max(results['rmse']):.6f} (范围={rmse_range:.6f})")
        
        if ssim_range < 0.02:
            print(f"\n💡 结论: SSIM变化<2%，说明K≥{steps[1]}时性能已饱和！")
        
        print("="*80)

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
    
    # ===== 检查样本是否在测试集中（默认总是检查，除非用户明确跳过） =====
    if not args.skip_test_check:  # 改为：默认检查，只有加了 --skip_test_check 才跳过
        import json
        test_samples_path = 'fm/ASPP/saved_models/test_samples.json'
        if os.path.exists(test_samples_path):
            with open(test_samples_path, 'r') as f:
                test_samples = json.load(f)
            
            current_sample = [args.map_id, args.tx_id]
            if current_sample in test_samples:
                print(f"✅ 样本 (map_id={args.map_id}, tx_id={args.tx_id}) 在测试集中")
                print(f"   这是真正的泛化测试，结果可信！\n")
            else:
                print(f"⚠️ 警告：样本 (map_id={args.map_id}, tx_id={args.tx_id}) 在训练集中！")
                print(f"   模型在训练时见过这个样本，不是真正的泛化测试！")
                print(f"   建议选择测试集中的样本进行评估。\n")
        else:
            print(f"ℹ️ 未找到测试集索引文件: {test_samples_path}")
            print(f"   首次运行或训练前生成的模型无法验证样本来源。")
            print(f"   建议运行 main_aspp.py 重新训练以生成测试集索引。\n")
    # ====================================================================
    
    # 初始化
    if not os.path.exists(args.model_path):
        print("⚠️ 模型文件不存在，请先运行 main.py 训练模型。")
        # 实际推理需要模型，直接退出
        return
        
    try:
        engine = InferenceEngine(args.model_path)
    except Exception as e:
        # 统一在主流程处理加载失败
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载数据
    try:
        cond, bld, tx, real = engine.load_condition_data(args.data_dir, args.map_id, args.tx_id)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 如果启用了步数对比模式
    if args.compare_steps:
        # 解析步数列表
        step_list = [int(s.strip()) for s in args.step_list.split(',')]
        print(f"\n🔬 步数对比模式")
        print(f"   测试步数: {step_list}")
        print(f"   样本: map_id={args.map_id}, tx_id={args.tx_id}")
        
        # 执行对比测试
        results = engine.compare_different_steps(cond, real, step_list=step_list, 
                                                 save_path='fm/ASPP/image/step_comparison.png')
        return

    # 原有的单步数测试流程
    # 生成
    print(f"🎨 生成中 (Steps={args.steps})...")
    import time
    
    # 预热GPU（首次运行会有CUDA kernel编译开销）
    print("🔥 预热GPU中...")
    _ = engine.generate(cond, steps=args.steps, profile=False)
    
    # 正式测试（重复3次取平均值）
    print("⏱️  正式测试（3次平均）...")
    times = []
    for i in range(3):
        _, perf = engine.generate(cond, steps=args.steps, profile=True)
        times.append(perf['total'])
        print(f"   第{i+1}次: {perf['total']:.4f}s")
    
    avg_time = sum(times) / len(times)
    
    # 最后一次生成用于可视化
    gen, perf = engine.generate(cond, steps=args.steps, profile=True)
    
    print(f"\n📊 性能统计:")
    print(f"   平均GPU计算时间: {avg_time:.4f}s")
    print(f"   最小时间: {min(times):.4f}s")
    print(f"   最大时间: {max(times):.4f}s")
    print(f"   (已预热，排除首次CUDA启动开销)")
    
    # 可视化
    engine.visualize_result(bld, tx, gen, real, save_path='fm/ASPP/image/fm_aspp_result.png')

if __name__ == "__main__":
    main()
