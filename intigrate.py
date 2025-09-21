from PIL import Image
import os
import numpy as np
import math

def infer_grid_size(num_patches):
    """
    根据子图片数量推断网格的行数和列数，优先选择宽布局（列数 >= 行数）。
    """
    sqrt_num = math.ceil(math.sqrt(num_patches))
    for cols in range(sqrt_num, num_patches + 1):
        if num_patches % cols == 0:
            rows = num_patches // cols
            # 优先选择列数 >= 行数的布局（例如 3x4 而不是 4x3）
            if cols >= rows:
                return rows, cols
    cols = sqrt_num
    rows = math.ceil(num_patches / cols)
    return rows, cols

def merge_images(input_folder, output_path, original_size=None, patch_size=256, overlap=64):
    """
    将子图片合并成一张大图。
    
    参数：
    input_folder: 子图片所在的文件夹
    output_path: 合并后图片的保存路径
    original_size: 原始图片的尺寸 (width, height)，如果提供则裁剪到此尺寸
    patch_size: 子图片的尺寸（默认 256）
    overlap: 分割时的重叠像素（默认 64）
    """
    # 获取子图片列表，适配 img2patch_ 前缀
    patch_files = sorted([f for f in os.listdir(input_folder) if f.startswith('img6patch_') and f.endswith('.png')],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    num_patches = len(patch_files)
    if num_patches == 0:
        print("错误：文件夹中没有找到子图片！")
        return
    
    # 推断网格大小
    grid_rows, grid_cols = infer_grid_size(num_patches)
    print(f"推断网格大小：{grid_rows} 行 x {grid_cols} 列")
    
    # 计算填充后图片的尺寸
    step = patch_size - overlap  # 步长考虑重叠
    target_width = grid_cols * step + overlap
    target_height = grid_rows * step + overlap
    print(f"推断填充后尺寸：{target_width}x{target_height}")
    
    # 创建空白画布和累加数组（使用 float32）
    merged_image_array = np.zeros((target_height, target_width, 3), dtype=np.float32)
    count_map = np.zeros((target_height, target_width, 3), dtype=np.float32)
    
    # 遍历子图片
    for i, patch_file in enumerate(patch_files):
        patch_path = os.path.join(input_folder, patch_file)
        patch = Image.open(patch_path).convert('RGB')
        
        # 确保子图片尺寸正确
        if patch.size != (patch_size, patch_size):
            print(f"警告：子图片 {patch_file} 尺寸 {patch.size} 与预期 {patch_size}x{patch_size} 不符")
            patch = patch.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        
        # 计算子图片在网格中的位置（按行优先）
        row = i // grid_cols
        col = i % grid_cols
        x_start = col * step
        y_start = row * step
        x_end = x_start + patch_size
        y_end = y_start + patch_size
        
        # 转换为 numpy 数组
        patch_array = np.array(patch, dtype=np.float32)
        
        # 粘贴到合并图像
        for y in range(y_start, min(y_end, target_height)):
            for x in range(x_start, min(x_end, target_width)):
                patch_y = y - y_start
                patch_x = x - x_start
                if patch_y < patch_array.shape[0] and patch_x < patch_array.shape[1]:
                    merged_image_array[y, x] += patch_array[patch_y, patch_x]
                    count_map[y, x] += 1
    
    # 处理重叠区域（取平均值）
    count_map[count_map == 0] = 1  # 避免除以零
    merged_image_array = merged_image_array / count_map
    merged_image_array = np.clip(merged_image_array, 0, 255).astype(np.uint8)
    merged_image = Image.fromarray(merged_image_array)
    
    # 如果提供了原始尺寸，裁剪掉填充区域
    if original_size:
        original_width, original_height = original_size
        x1 = (target_width - original_width) // 2
        y1 = (target_height - original_height) // 2
        x2 = x1 + original_width
        y2 = y1 + original_height
        merged_image = merged_image.crop((x1, y1, x2, y2))
    
    # 保存合并后的图片
    merged_image.save(output_path)
    print(f"合并完成，保存至 {output_path}")

# 示例用法
if __name__ == "__main__":
    # 配置参数
    input_folder = r"C:\Users\27805\Desktop\bin\test\patches"  # 子图片所在文件夹
    output_path = r"C:\Users\27805\Desktop\bin\test\ourenhanced_image6.png"  # 合并后图片保存路径
    original_size = None  # 替换为原始图片的尺寸（例如 (768, 768)），如果不知道设为 None
    
    merge_images(input_folder, output_path, original_size)