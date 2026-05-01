"""这是深度图转换点云的函数"""

__author__ = "Hsiylin"

import cv2
import numpy as np
import open3d as o3d

# 剔除异常数据并滤波降噪
def preprocess_depth_map(depth_map, min_range=0.1, max_range=50.0):

    # 异常值剔除
    processed_depth = np.where((depth_map < min_range) | (depth_map > max_range), 0, depth_map)
    
    # 中值滤波
    processed_depth = cv2.medianBlur(processed_depth.astype(np.float32), 3)

    # 双边滤波
    processed_depth = cv2.bilateralFilter(processed_depth, d=5, sigmaColor=0.05, sigmaSpace=5)
    
    return processed_depth

def generate_pcd(image_path, fx = 500.0, fy = 500.0, cx = 100, cy = 100):

    depth_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 如果是彩色图则转化为灰度图
    if len(depth_data.shape) > 2:
        depth_data = cv2.cvtColor(depth_data, cv2.COLOR_BGR2GRAY)
    
    # 确保深度数据是 uint16 或 float32 类型，否则o3d无法转换成点云
    if depth_data.dtype != np.uint16 and depth_data.dtype != np.float32:
        depth_data = depth_data.astype(np.uint16) 

    # 预处理深度图
    depth_data = preprocess_depth_map(depth_data)
        
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