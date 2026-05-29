"""这是深度图转换点云的函数"""

__author__ = "Hsiylin"

import cv2
import numpy as np
import open3d as o3d

# 剔除异常数据并滤波降噪
def preprocess_depth_map(depth_map, min_range=0.1, max_range=65536):

    # 异常值剔除
    processed_depth = np.where((depth_map < min_range) | (depth_map > max_range), 0, depth_map)
    
    valid_mask = (processed_depth > 0)

    # 中值滤波
    processed_depth = cv2.medianBlur(processed_depth.astype(np.float32), 3)

    # 双边滤波
    processed_depth = cv2.bilateralFilter(processed_depth, d=5, sigmaColor=0.05, sigmaSpace=5)
    
    final_depth = np.where(valid_mask, processed_depth, 0.0)

    return final_depth.astype(np.float32)

def adaptive_extract_depth(img, min_depth=0.0, max_depth=1.0):
    # 单通道，灰度图
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return img.astype(np.float32) / 255.0
    
    # 三通道
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 检查RGB通道
        channel_diff = np.max(np.abs(img[:,:,0].astype(int) - img[:,:,1].astype(int)))
        
        if channel_diff < 5:  
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray.astype(np.float32) / 255.0
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:, :, 0].astype(np.float32)
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            valid_color = (s_channel > 30) & (v_channel > 30)
            
            h_normalized = np.clip(h_channel, 0, 180) / 180.0
            depth_recovered = h_normalized * (max_depth - min_depth) + min_depth
            
            # 将黑白背景区域强制归零
            depth_recovered = np.where(valid_color, depth_recovered, 0.0)
            return depth_recovered

def generate_pcd(image_path, fx = 500.0, fy = 500.0, cx = 100, cy = 100):

    depth_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    depth_data = adaptive_extract_depth(depth_data)
    
    '''
    print("--- 滤波前深度图状态检查 ---")
    print(f"数据类型 (dtype): {depth_data.dtype}")
    print(f"矩阵形状 (shape): {depth_data.shape}")
    print(f"最大深度值 (max): {np.max(depth_data)}")
    print(f"最小深度值 (min): {np.min(depth_data)}")
    # 计算图像中有多少个像素点的值大于 0
    non_zero_count = np.sum(depth_data > 0)
    print(f"有效大于0的像素点数量: {non_zero_count}")
    print("--------------------------------")
    '''


    # 预处理深度图
    depth_data = preprocess_depth_map(depth_data,0,2)
        
    o3d_depth = o3d.geometry.Image(depth_data)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_data.shape[1], depth_data.shape[0], fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, intrinsic)

    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    
    return pcd

if __name__ == "__main__":
    pcd = generate_pcd("savedata/test1.png")
    o3d.visualization.draw_geometries([pcd], window_name="生成的点云", width=800, height=600)