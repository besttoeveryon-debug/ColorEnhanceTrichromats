from PIL import Image
import os
import math
import matplotlib.pyplot as plt

def pad_image(image, patch_size=256, overlap=64):
    """将图像填充到基于步长的尺寸"""
    width, height = image.size
    step = patch_size - overlap  # 192
    # 计算需要的列数和行数
    cols = math.ceil((width - patch_size) / step) + 1  # ceil((1001-256)/192)+1=5
    rows = math.ceil((height - patch_size) / step) + 1  # ceil((565-256)/192)+1=3
    # 填充尺寸
    target_width = (cols - 1) * step + patch_size  # 4*192 + 256 = 1024
    target_height = (rows - 1) * step + patch_size  # 2*192 + 256 = 640
    
    padded = Image.new('RGB', (target_width, target_height), (255, 255, 255))
    x_offset = (target_width - width) // 2  # (1024-1001)//2=11
    y_offset = (target_height - height) // 2  # (640-565)//2=37
    padded.paste(image, (x_offset, y_offset))
    
    return padded, (x_offset, y_offset, x_offset + width, y_offset + height), (target_width, target_height)

def split_image(image, target_size, original_coords, patch_size=256, overlap=64):
    """分割图像，考虑重叠，取消 overlap_area"""
    target_width, target_height = target_size
    patches = []
    coords = []
    step = patch_size - overlap  # 192
    
    grid_cols = math.ceil((target_width - patch_size) / step) + 1  # 5
    grid_rows = math.ceil((target_height - patch_size) / step) + 1  # 3
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            x_start = j * step
            y_start = i * step
            x_end = min(x_start + patch_size, target_width)
            y_end = min(y_start + patch_size, target_height)
            
            patch = image.crop((x_start, y_start, x_end, y_end))
            if patch.size != (patch_size, patch_size):
                patch = patch.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
            
            patches.append(patch)
            coords.append((x_start, y_start, x_end, y_end))
    
    return patches, coords, (grid_rows, grid_cols)

def split_and_save_image(image_path, output_dir, patch_size=256, overlap=64):
    """主函数：填充并分割图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        padded_image, original_coords, target_size = pad_image(image, patch_size, overlap)
        print(f"原始尺寸: {image.size}, 填充后尺寸: {target_size[0]}x{target_size[1]}")
        print(f"原始图像坐标: {original_coords}")
        
        patches, coords, (grid_rows, grid_cols) = split_image(padded_image, target_size, original_coords, patch_size, overlap)
        
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*4, grid_rows*4))
        axes = axes.flatten() if grid_rows * grid_cols > 1 else [axes]
        
        for ax in axes:
            ax.axis('off')
        
        for i, (patch, (x_start, y_start, x_end, y_end)) in enumerate(zip(patches, coords)):
            patch_path = os.path.join(output_dir, f"img3patch_{i+1}.png")
            patch.save(patch_path)
            print(f"子图 {i+1} 已保存到 {patch_path}")
            
            ax = axes[i]
            ax.imshow(patch)
            ax.set_title(f"子图 {i+1}\n({x_start},{y_start})-({x_end},{y_end})", fontsize=8)
            ax.axis('off')
        
        vis_path = os.path.join(output_dir, "patches_vis.png")
        plt.tight_layout()
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"子图可视化已保存到 {vis_path}")
        
        return patches, coords, original_coords, target_size
    
    except Exception as e:
        print(f"运行时发生错误: {e}")
        return None, None, None, None

if __name__ == "__main__":
    input_image = "C:\Users\27805\Desktop\微信图片_20250725222051.png"
    output_dir = r"C:\Users\27805\Desktop\bin\test\patches"
    patches, coords, original_coords, target_size = split_and_save_image(input_image, output_dir)
    if patches:
        print(f"分割坐标: {coords}")
        print(f"原始图像坐标: {original_coords}")
        print(f"目标尺寸: {target_size[0]}x{target_size[1]}")
        print(f"分割成 {len(patches)} 份")