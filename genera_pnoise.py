"""这是一个生成perlin噪声深度图的脚本, 与软件无关, 但可以用来生成测试用的深度图"""

__author__ = "Hsiylin"

import numpy as np
import cv2
from noise import pnoise2
from pathlib import Path

def generate_perlin_noise(folder_name = "savedata"):

    # 确保目录存在
    save_dir = Path(__file__).parent / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存图像的函数，防止覆盖已有文件
    def save_image(image,folder = save_dir, base_name="image"):
        user_input = input(f"请输入文件名 (直接回车使用默认名{base_name}): ").strip()
        if user_input:
            base_name = user_input
    
        save_path = folder / f"{base_name}.png"
        counter = 1
        while save_path.exists():
            save_path = folder / f"{base_name}_{counter}.png"
            counter += 1
        cv2.imwrite(str(save_path), image)



    width = int(input("请输入图像宽度: "))
    height = int(input("请输入图像高度: "))

    scale = float(input("请输入噪声缩放因子 (0-100): "))

    octaves = int(input("请输入噪声的倍频 (1-10): "))

    persistence =float(input("请输入噪声的持久性 (0-1): "))

    lacunarity = int(input("请输入噪声的频率增益 (1-4): "))

    random_depth = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # 生成 Perlin 噪声图
    depth_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            depth_map[y][x] = pnoise2(  x / scale, 
                                        y / scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity,
                                        repeatx=width,
                                        repeaty=height,
                                        base=1914)
        
    # 将噪声值归一化到 0-255
    depth_map_rescaled = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 生成彩色图像
    color_depth = cv2.applyColorMap(depth_map_rescaled, cv2.COLORMAP_JET)

    # 保存图像
    save_image(depth_map_rescaled, base_name="perlin_noise")
    save_image(color_depth, base_name="perlin_noise_colormap")

if __name__ == "__main__":
    generate_perlin_noise()
    