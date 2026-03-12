__author__ = "Hsiylin"

import os
import sys

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import cv2
import open3d as o3d
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QFileDialog, QGroupBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from genera_o3d import generate_pcd

class Depth3DPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("深度图可视化 (Linux/Win)")
        self.resize(600, 350)
        
        self.img_path = None #深度图路径
        self.raw_img = None #原始图像（可能是彩色图）
        self.depth_map = None #深度图数据（灰度图）
        self.pcd = None #生成的点云对象

        # 初始化 Open3D 独立窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Viewer", width=800, height=600)
        
        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_vis)
        self.timer.start(30)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 深度图预览区
        self.preview_area = QVBoxLayout()
        self.path_label = QLabel("未选择文件")
        self.path_label.setWordWrap(True)
        self.img_label = QLabel("等待导入深度图...")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: 2px dashed #aaa; background: #f0f0f0;")
        self.img_label.setFixedSize(400, 300)
        
        self.preview_area.addWidget(self.path_label)
        self.preview_area.addWidget(self.img_label)
        main_layout.addLayout(self.preview_area, 2)

        # 控制面板
        ctrl_panel = QVBoxLayout()
        
        # 相机内参组
        param_group = QGroupBox("相机内参设置")
        param_layout = QVBoxLayout()
        self.in_fx = QLineEdit(); self.in_fx.setPlaceholderText("Fx (默认500.0)")
        self.in_fy = QLineEdit(); self.in_fy.setPlaceholderText("Fy (默认500.0)")
        self.in_cx = QLineEdit(); self.in_cx.setPlaceholderText("Cx (默认图像宽度/2)")
        self.in_cy = QLineEdit(); self.in_cy.setPlaceholderText("Cy (默认图像高度/2)")
        for w in [self.in_fx, self.in_fy, self.in_cx, self.in_cy]: param_layout.addWidget(w)
        param_group.setLayout(param_layout)
        ctrl_panel.addWidget(param_group)

        # 读取深度图按钮
        self.btn_open = QPushButton("打开深度图")
        self.btn_open.setMinimumHeight(50)
        self.btn_open.clicked.connect(self.open_file)
        
        # 生成/刷新点云按钮
        self.btn_main = QPushButton("生成点云") 
        self.btn_main.setMinimumHeight(50)
        self.btn_main.clicked.connect(self.handle_main_action)
        self.btn_main.setEnabled(False)

        # 保存点云按钮
        self.btn_save = QPushButton("保存点云(.ply)")
        self.btn_save.setMinimumHeight(50)
        self.btn_save.clicked.connect(self.save_model)
        self.btn_save.setEnabled(False)
        
        # 添加按钮到控制面板
        ctrl_panel.addWidget(self.btn_open)
        ctrl_panel.addWidget(self.btn_main)
        ctrl_panel.addWidget(self.btn_save)
        ctrl_panel.addStretch()
        
        main_layout.addLayout(ctrl_panel, 1)

    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_render()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择深度图", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.img_path = path
            self.path_label.setText(f"路径: {path}")
            self.depth_map = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.raw_img = cv2.imread(path)  # 可能是彩色图

            # 显示缩略图
            qt_img = QImage(self.raw_img.data, self.raw_img.shape[1], 
                            self.raw_img.shape[0], self.raw_img.shape[1]*3, QImage.Format_BGR888)
            self.img_label.setPixmap(QPixmap.fromImage(qt_img).scaled(400, 300, Qt.KeepAspectRatio))
            
            self.update_button_state()

    def get_current_params(self):
        # 获取当前 UI 上的所有参数状态
        return (self.img_path, self.in_fx.text(), self.in_fy.text(), self.in_cx.text(), self.in_cy.text())

    def update_button_state(self):
        # 控制点云生成或刷新的按钮
        if self.img_path is None:
            self.btn_main.setEnabled(False)
            self.btn_save.setEnabled(False)
            return

        self.btn_main.setEnabled(True)
        current = self.get_current_params()

        if self.pcd is None:
            self.btn_main.setText("生成点云")
            self.btn_main.setStyleSheet("background-color: #e1f5fe;")
            self.btn_save.setEnabled(False)
        else:
            self.btn_main.setText("刷新点云")
            self.btn_main.setStyleSheet("background-color: #fff9c4;")
            self.btn_save.setEnabled(True)

    def handle_main_action(self):
        self.pcd_generator()
        self.update_button_state()

    def pcd_generator(self):
        # 深度图转点云
        h, w = self.depth_map.shape
        
        # 相机内参，为空则使用默认
        try:
            fx = float(self.in_fx.text()) if self.in_fx.text() else 500.0
            fy = float(self.in_fy.text()) if self.in_fy.text() else 500.0
            cx = float(self.in_cx.text()) if self.in_cx.text() else w/2
            cy = float(self.in_cy.text()) if self.in_cy.text() else h/2
        except:
            QMessageBox.warning(self, "错误", "内参格式不正确，请输入数字")
            return

        self.pcd = generate_pcd(self.img_path, fx, fy, cx, cy)


        # 更新渲染
        self.vis.clear_geometries()
        self.vis.add_geometry(self.pcd)
        self.vis.reset_view_point(True)

        self.update_button_state()



    def save_model(self):
        if self.pcd is None: return
        path, _ = QFileDialog.getSaveFileName(self, "保存点云", "point_cloud.ply", "Point Cloud Files (*.ply);;All Files (*)")
        if path:
            o3d.io.write_point_cloud(path, self.pcd)
            QMessageBox.information(self, "成功", f"点云已保存至:\n{path}")

    def closeEvent(self, event):
        self.vis.destroy_window()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = Depth3DPro()
    gui.show()
    sys.exit(app.exec_())
