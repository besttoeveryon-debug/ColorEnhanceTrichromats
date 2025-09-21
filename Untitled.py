import os
# 在导入其他库之前设置环境变量以避免 OMP 错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免多线程问题
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as utils

def safe_imshow(ax, tensor, title):
    """安全显示图像，确保数据范围和维度正确"""
    np_img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    np_img = np.clip(np_img, 0, 1)  # 裁剪到 [0, 1]
    ax.imshow(np_img)
    ax.set_title(title)
    ax.axis('off')

def create_scharr_filters(device):
    """创建 Scharr 滤波器"""
    scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).to(device)
    scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).to(device)
    return scharr_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1), scharr_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

def multi_scale_scharr_edges(image_tensor, scales=[1.0, 0.5, 0.25], device=None):
    """多尺度 Scharr 边缘检测，优化细小纹理"""
    if device is None:
        device = image_tensor.device
    
    scharr_x, scharr_y = create_scharr_filters(device)
    all_edges = []
    
    for scale in scales:
        scaled_img = F.interpolate(image_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
        edge_x = F.conv2d(scaled_img, scharr_x, padding=1, groups=3)
        edge_y = F.conv2d(scaled_img, scharr_y, padding=1, groups=3)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        edge_mag = F.interpolate(edge_mag, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
        all_edges.append(edge_mag)
    
    combined_edges = torch.max(torch.stack(all_edges), dim=0)[0]
    return combined_edges

def enhance_weak_textures(img_tensor, kernel_size=9, clip_limit=0.03):
    """弱纹理增强，优化羽毛等细小纹理"""
    # 局部对比度增强
    local_mean = F.avg_pool2d(img_tensor, kernel_size, stride=1, padding=kernel_size//2)
    local_var = F.avg_pool2d(img_tensor**2, kernel_size, stride=1, padding=kernel_size//2) - local_mean**2
    local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))
    
    enhanced = (img_tensor - local_mean) / (local_std + 1e-6)
    enhanced = torch.clamp(enhanced * clip_limit + 0.5, 0, 1)
    
    # 高频强调
    blurred = F.avg_pool2d(enhanced, 3, stride=1, padding=1)
    high_pass = enhanced - blurred
    return torch.clamp(enhanced + high_pass * 1.2, 0, 1)  # 提高高频增强权重

def visualize_edges(image_path, output_path, size=(512, 512)):
    """提取纹理并可视化，优化边缘图亮度"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 弱纹理增强
        enhanced_img = enhance_weak_textures(img_tensor)
        
        # 多尺度边缘检测
        edges = multi_scale_scharr_edges(enhanced_img, device=device)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 显示原始图像和增强图像
        safe_imshow(axes[0], img_tensor, "原始图像")
        safe_imshow(axes[1], enhanced_img, "纹理增强")
        
        # 边缘图处理：对数变换增强亮度
        edge_np = edges.squeeze(0).permute(1, 2, 0).mean(dim=2).cpu().detach().numpy()
        edge_np = np.log1p(edge_np) / np.log1p(edge_np.max())  # 对数变换提高弱纹理亮度
        edge_np = np.clip(edge_np, 0, 1)  # 确保范围
        axes[2].imshow(edge_np, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title("多尺度边缘 (Scharr)")
        axes[2].axis('off')
        
        # 保存可视化结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"可视化结果已保存到 {output_path}")
        
        # 保存边缘图
        edge_path = output_path.replace(".jpg", "_edges.png")
        edges_normalized = torch.tensor(edge_np).unsqueeze(0)
        utils.save_image(edges_normalized, edge_path)
        print(f"纯边缘图已保存到 {edge_path}")
        
        # 计算边缘强度
        edge_intensity = edges.mean().item()
        print(f"边缘强度: {edge_intensity:.4f}（值越高表示纹理越丰富）")
        
        plt.close(fig)
        return edges
    
    except Exception as e:
        print(f"运行时发生错误: {e}")
        return None

if __name__ == "__main__":
    input_image = r"C:\Users\27805\Desktop\bin\test\511.jpg"
    output_image = r"C:\Users\27805\Desktop\bin\test\edge_visualization.jpg"
    edge_map = visualize_edges(input_image, output_image)