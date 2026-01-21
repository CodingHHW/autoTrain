#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import tempfile
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QGroupBox, QLabel, QLineEdit, QFileDialog, QSlider, 
    QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit, QMessageBox,
    QSplitter, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from loguru import logger

# 配置日志
logger.add("logs/app.log", rotation="500 MB", level="INFO", encoding="utf-8")

# 获取默认保存路径

def get_default_save_path(subdir=None):
    """
    根据操作系统类型获取默认保存路径
    - macOS/Linux: 系统临时目录
    - Windows: D盘根目录
    """
    platform = sys.platform
    
    if platform.startswith('win'):
        # Windows系统，使用D盘根目录
        base_path = "D:/"
    else:
        # macOS或Linux系统，使用系统临时目录
        base_path = tempfile.gettempdir()
    
    # 如果指定了子目录，则拼接完整路径
    if subdir:
        base_path = os.path.join(base_path, subdir)
    
    return base_path

# 检查并创建路径
def ensure_path_exists(path):
    """
    检查路径是否存在，如果不存在则创建
    如果路径不可写则抛出异常
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"创建目录: {path}")
        
        # 检查路径是否可写
        if not os.access(path, os.W_OK):
            raise PermissionError(f"路径不可写: {path}")
        
        return True
    except Exception as e:
        logger.error(f"处理路径 {path} 时出错: {e}")
        raise

class VideoRecorderThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal()
    
    def __init__(self, fps=25, save_path="./videos", width=None, height=None):
        super().__init__()
        self.fps = fps
        self.save_path = save_path
        self.width = width
        self.height = height
        self.is_recording = False
        self.is_paused = False
        self.cap = None
        self.out = None
    
    def run(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("无法打开摄像头")
                return
            
            # 获取摄像头分辨率，如果用户指定了分辨率则使用用户指定的值
            cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 使用用户指定的分辨率或摄像头默认分辨率
            width = self.width if self.width is not None else cam_width
            height = self.height if self.height is not None else cam_height
            
            # 如果用户指定了分辨率，设置摄像头分辨率
            if self.width is not None and self.height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 检查并创建保存目录
            ensure_path_exists(self.save_path)
            
            # 生成文件名
            # 使用时间戳生成文件名，格式：YYYYMMDD_HHMMSS.mp4
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_path, f"record_{timestamp}.mp4")
            
            # 创建视频写入对象
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
            
            logger.info(f"开始录制视频，保存路径：{filename}")
            
            while self.is_recording:
                if not self.is_paused:
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_signal.emit(frame)
                        self.out.write(frame)
            
            self.finished_signal.emit()
            logger.info("视频录制结束")
            
        except Exception as e:
            logger.error(f"录制视频时发生错误：{e}")
        finally:
            if self.out:
                self.out.release()
            if self.cap:
                self.cap.release()
    
    def start_recording(self):
        self.is_recording = True
        self.is_paused = False
        self.start()
    
    def pause_recording(self):
        self.is_paused = not self.is_paused
    
    def stop_recording(self):
        self.is_recording = False

class ImageExtractorThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, video_path, save_path, num_images):
        super().__init__()
        self.video_path = video_path
        self.save_path = save_path
        self.num_images = num_images
    
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件：{self.video_path}")
                self.finished_signal.emit("错误：无法打开视频文件")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.num_images:
                logger.warning(f"视频帧数不足，仅能提取{total_frames}张图像")
                self.num_images = total_frames
            
            # 检查并创建保存目录
            ensure_path_exists(self.save_path)
            
            # 随机选择帧
            frame_indices = np.random.choice(total_frames, self.num_images, replace=False)
            frame_indices.sort()
            
            current_frame = 0
            extracted_count = 0
            
            while current_frame < total_frames and extracted_count < self.num_images:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame in frame_indices:
                    image_path = os.path.join(self.save_path, f"frame_{current_frame}.jpg")
                    cv2.imwrite(image_path, frame)
                    extracted_count += 1
                    progress = int((extracted_count / self.num_images) * 100)
                    self.progress_signal.emit(progress)
                
                current_frame += 1
            
            cap.release()
            self.finished_signal.emit(f"成功提取{extracted_count}张图像")
            logger.info(f"成功从视频中提取{extracted_count}张图像，保存路径：{self.save_path}")
            
        except Exception as e:
            logger.error(f"提取图像时发生错误：{e}")
            self.finished_signal.emit(f"错误：{str(e)}")

class DatasetSplitterThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, images_dir, labels_dir, output_dir, train_ratio, val_ratio, test_ratio):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def run(self):
        try:
            import shutil
            
            # 清空输出目录下的images和labels文件夹，避免训练数据被污染
            logger.info(f"开始清空输出目录：{self.output_dir}")
            
            # 清空images目录
            images_dir = os.path.join(self.output_dir, "images")
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
                logger.info(f"已清空目录：{images_dir}")
            
            # 清空labels目录
            labels_dir = os.path.join(self.output_dir, "labels")
            if os.path.exists(labels_dir):
                shutil.rmtree(labels_dir)
                logger.info(f"已清空目录：{labels_dir}")
            
            # 获取所有图像文件
            all_image_files = [f for f in os.listdir(self.images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
            if not all_image_files:
                self.finished_signal.emit("错误：未找到图像文件")
                return
            
            # 过滤掉没有对应标签文件的图像
            image_files = []
            for img_file in all_image_files:
                # 生成对应的标签文件名
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(self.labels_dir, label_file)
                # 只保留有对应标签文件的图像
                if os.path.exists(label_path):
                    image_files.append(img_file)
            
            if not image_files:
                self.finished_signal.emit("错误：未找到有对应标签的图像文件")
                return
            
            # 打乱顺序
            np.random.shuffle(image_files)
            
            total = len(image_files)
            train_count = int(total * self.train_ratio / 100)
            val_count = int(total * self.val_ratio / 100)
            test_count = total - train_count - val_count
            
            # 创建输出目录结构
            for split in ["train", "val", "test"]:
                os.makedirs(os.path.join(self.output_dir, "images", split), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, "labels", split), exist_ok=True)
            
            logger.info(f"输出目录结构创建完成：{self.output_dir}")
            
            # 划分数据集
            splits = {
                "train": image_files[:train_count],
                "val": image_files[train_count:train_count+val_count],
                "test": image_files[train_count+val_count:]
            }
            
            # 复制文件
            processed = 0
            for split_name, files in splits.items():
                for file in files:
                    # 复制图像文件
                    src_img = os.path.join(self.images_dir, file)
                    dst_img = os.path.join(self.output_dir, "images", split_name, file)
                    shutil.copy(src_img, dst_img)
                    
                    # 复制标签文件
                    label_file = os.path.splitext(file)[0] + ".txt"
                    src_label = os.path.join(self.labels_dir, label_file)
                    if os.path.exists(src_label):
                        dst_label = os.path.join(self.output_dir, "labels", split_name, label_file)
                        shutil.copy(src_label, dst_label)
                    
                    processed += 1
                    progress = int((processed / total) * 100)
                    self.progress_signal.emit(progress)
            
            # 生成报告
            report = f"数据集划分完成！\n"
            report += f"总图像数：{total}\n"
            report += f"训练集：{train_count}张 ({self.train_ratio}%)\n"
            report += f"验证集：{val_count}张 ({self.val_ratio}%)\n"
            report += f"测试集：{test_count}张 ({self.test_ratio}%)\n"
            report += f"输出目录：{self.output_dir}"
            
            self.finished_signal.emit(report)
            logger.info(report)
            
        except Exception as e:
            logger.error(f"划分数据集时发生错误：{e}")
            self.finished_signal.emit(f"错误：{str(e)}")

class YOLO_trainerThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, data_file, epochs=50, batch_size=16, lr=0.001, model_config="yolov11n.yaml", pretrained_weights="yolov11n.pt"):
        super().__init__()
        self.data_file = data_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_config = model_config
        self.pretrained_weights = pretrained_weights
        self.is_running = False
    
    def run(self):
        try:
            from ultralytics import YOLO
            
            self.is_running = True
            logger.info(f"开始YOLO训练，数据文件：{self.data_file}")
            
            # 加载模型
            model = YOLO(self.model_config)
            if os.path.exists(self.pretrained_weights):
                model = YOLO(self.pretrained_weights)
            
            # 训练模型
            results = model.train(
                data=self.data_file,
                epochs=self.epochs,
                batch=self.batch_size,
                lr0=self.lr,
                verbose=True
            )
            
            self.output_signal.emit(f"训练完成！结果：{results}")
            logger.info("YOLO训练完成")
            
        except Exception as e:
            logger.error(f"YOLO训练时发生错误：{e}")
            self.output_signal.emit(f"训练错误：{str(e)}")
        finally:
            self.is_running = False
            self.finished_signal.emit()
    
    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.video_recorder = None
        self.image_extractor = None
        self.dataset_splitter = None
        self.yolo_trainer = None
    
    def init_ui(self):
        self.setWindowTitle("视频处理与模型训练集成软件")
        # 设置固定窗口尺寸
        self.setGeometry(100, 100, 1200, 800)
        # 窗口状态标志
        self.is_fullscreen = False
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # 创建左侧工具栏
        self.toolbar = QWidget()
        toolbar_layout = QVBoxLayout(self.toolbar)
        
        # 创建视频录制模块
        self.create_video_recording_module(toolbar_layout)
        
        # 创建图像提取模块
        self.create_image_extraction_module(toolbar_layout)
        
        # 创建数据标注模块
        self.create_data_annotation_module(toolbar_layout)
        
        # 创建数据集切分模块
        self.create_dataset_splitting_module(toolbar_layout)
        
        # 创建YOLO训练模块
        self.create_yolo_training_module(toolbar_layout)
        
        # 创建右侧视频显示区域
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        
        # 视频显示标签
        self.video_label = QLabel("视频显示区域")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        
        # 设置视频标签的初始大小和大小策略，避免窗口自动调整
        self.video_label.setMinimumSize(640, 480)  # 设置最小大小
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展但不强制
        
        right_layout.addWidget(self.video_label)
        
        # 状态显示
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(150)
        right_layout.addWidget(self.status_text)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.toolbar)
        splitter.addWidget(self.right_widget)
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
    
    def create_video_recording_module(self, layout):
        group = QGroupBox("视频录制")
        group_layout = QVBoxLayout(group)
        
        # 录制视频按钮
        self.record_btn = QPushButton("录制视频")
        self.record_btn.clicked.connect(self.start_video_recording)
        group_layout.addWidget(self.record_btn)
        
        # 暂停/继续按钮
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_video_recording)
        self.pause_btn.setEnabled(False)
        group_layout.addWidget(self.pause_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_video_recording)
        self.stop_btn.setEnabled(False)
        group_layout.addWidget(self.stop_btn)
        
        # 参数配置
        param_layout = QVBoxLayout()
        
        # 帧率设置
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("帧率:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(5)  # 将默认帧率设置为5
        fps_layout.addWidget(self.fps_spin)
        param_layout.addLayout(fps_layout)
        
        # 分辨率设置
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("分辨率:"))
        self.resolution_combo = QComboBox()
        # 添加常见分辨率选项，格式："宽度x高度"，存储值："宽度,高度"
        self.resolution_combo.addItem("默认分辨率", "0,0")
        self.resolution_combo.addItem("640x480 (VGA)", "640,480")
        self.resolution_combo.addItem("1280x720 (720P)", "1280,720")
        self.resolution_combo.addItem("1920x1080 (1080P)", "1920,1080")
        self.resolution_combo.addItem("2560x1440 (2K)", "2560,1440")
        self.resolution_combo.addItem("3840x2160 (4K)", "3840,2160")
        resolution_layout.addWidget(self.resolution_combo)
        param_layout.addLayout(resolution_layout)
        
        # 保存路径
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("保存路径:"))
        self.video_save_path = QLineEdit(get_default_save_path("videos"))
        path_layout.addWidget(self.video_save_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_video_save_path)
        path_layout.addWidget(browse_btn)
        param_layout.addLayout(path_layout)
        
        group_layout.addLayout(param_layout)
        layout.addWidget(group)
    
    def create_image_extraction_module(self, layout):
        group = QGroupBox("图像提取")
        group_layout = QVBoxLayout(group)
        
        # 视频文件选择
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("视频文件:"))
        # 设置视频文件默认路径为视频录制的保存路径
        self.video_file_path = QLineEdit(get_default_save_path("videos"))
        video_layout.addWidget(self.video_file_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_video_file)
        video_layout.addWidget(browse_btn)
        group_layout.addLayout(video_layout)
        
        # 图像保存路径
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("保存路径:"))
        self.image_save_path = QLineEdit(get_default_save_path("images"))
        save_layout.addWidget(self.image_save_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_image_save_path)
        save_layout.addWidget(browse_btn)
        group_layout.addLayout(save_layout)
        
        # 提取数量
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("提取数量:"))
        self.image_count = QSpinBox()
        self.image_count.setRange(1, 1000)
        self.image_count.setValue(100)
        count_layout.addWidget(self.image_count)
        group_layout.addLayout(count_layout)
        
        # 提取按钮
        self.extract_btn = QPushButton("提取图像")
        self.extract_btn.clicked.connect(self.start_image_extraction)
        group_layout.addWidget(self.extract_btn)
        
        # 进度条
        self.extract_progress = QProgressBar()
        self.extract_progress.setValue(0)
        group_layout.addWidget(self.extract_progress)
        
        layout.addWidget(group)
    
    def create_data_annotation_module(self, layout):
        group = QGroupBox("数据标注")
        group_layout = QVBoxLayout(group)
        
        # 标注按钮
        self.annotate_btn = QPushButton("数据标注")
        self.annotate_btn.clicked.connect(self.start_data_annotation)
        group_layout.addWidget(self.annotate_btn)
        
        # 图像文件夹
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("图像文件夹:"))
        self.annotation_img_path = QLineEdit(get_default_save_path("images"))
        img_layout.addWidget(self.annotation_img_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_annotation_img_path)
        img_layout.addWidget(browse_btn)
        group_layout.addLayout(img_layout)
        
        # 标签保存路径
        lbl_layout = QHBoxLayout()
        lbl_layout.addWidget(QLabel("标签保存路径:"))
        self.annotation_lbl_path = QLineEdit(get_default_save_path("labels"))
        lbl_layout.addWidget(self.annotation_lbl_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_annotation_lbl_path)
        lbl_layout.addWidget(browse_btn)
        group_layout.addLayout(lbl_layout)
        
        layout.addWidget(group)
    
    def create_dataset_splitting_module(self, layout):
        group = QGroupBox("数据集切分")
        group_layout = QVBoxLayout(group)
        
        # 图像文件夹
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("图像文件夹:"))
        self.split_img_path = QLineEdit(get_default_save_path("images"))
        img_layout.addWidget(self.split_img_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_split_img_path)
        img_layout.addWidget(browse_btn)
        group_layout.addLayout(img_layout)
        
        # 标签文件夹
        lbl_layout = QHBoxLayout()
        lbl_layout.addWidget(QLabel("标签文件夹:"))
        self.split_lbl_path = QLineEdit(get_default_save_path("labels"))
        lbl_layout.addWidget(self.split_lbl_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_split_lbl_path)
        lbl_layout.addWidget(browse_btn)
        group_layout.addLayout(lbl_layout)
        
        # 输出目录
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("输出目录:"))
        self.split_output_path = QLineEdit(get_default_save_path("dataset"))
        out_layout.addWidget(self.split_output_path)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_split_output_path)
        out_layout.addWidget(browse_btn)
        group_layout.addLayout(out_layout)
        
        # 划分比例
        ratio_layout = QVBoxLayout()
        ratio_layout.addWidget(QLabel("划分比例 (训练:验证:测试):"))
        
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("训练集:"))
        self.train_ratio = QSpinBox()
        self.train_ratio.setRange(0, 100)
        self.train_ratio.setValue(70)
        train_layout.addWidget(self.train_ratio)
        train_layout.addWidget(QLabel("%"))
        ratio_layout.addLayout(train_layout)
        
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("验证集:"))
        self.val_ratio = QSpinBox()
        self.val_ratio.setRange(0, 100)
        self.val_ratio.setValue(20)
        val_layout.addWidget(self.val_ratio)
        val_layout.addWidget(QLabel("%"))
        ratio_layout.addLayout(val_layout)
        
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("测试集:"))
        self.test_ratio = QSpinBox()
        self.test_ratio.setRange(0, 100)
        self.test_ratio.setValue(10)
        test_layout.addWidget(self.test_ratio)
        test_layout.addWidget(QLabel("%"))
        ratio_layout.addLayout(test_layout)
        
        group_layout.addLayout(ratio_layout)
        
        # 切分按钮
        self.split_btn = QPushButton("数据集切分")
        self.split_btn.clicked.connect(self.start_dataset_splitting)
        group_layout.addWidget(self.split_btn)
        
        # 进度条
        self.split_progress = QProgressBar()
        self.split_progress.setValue(0)
        group_layout.addWidget(self.split_progress)
        
        layout.addWidget(group)
    
    def create_yolo_training_module(self, layout):
        group = QGroupBox("YOLO训练")
        group_layout = QVBoxLayout(group)
        
        # 只保留一个启动训练的按钮
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_yolo_training)
        group_layout.addWidget(self.train_btn)
        
        layout.addWidget(group)
    
    def browse_video_save_path(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.video_save_path.text()
        path = QFileDialog.getExistingDirectory(self, "选择视频保存路径", current_path)
        if path:
            self.video_save_path.setText(path)
    
    def browse_video_file(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.video_file_path.text()
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", current_path, "视频文件 (*.mp4 *.avi *.mov)")
        if path:
            self.video_file_path.setText(path)
    
    def browse_image_save_path(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.image_save_path.text()
        path = QFileDialog.getExistingDirectory(self, "选择图像保存路径", current_path)
        if path:
            self.image_save_path.setText(path)
    
    def browse_annotation_img_path(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.annotation_img_path.text()
        path = QFileDialog.getExistingDirectory(self, "选择图像文件夹", current_path)
        if path:
            self.annotation_img_path.setText(path)
    
    def browse_annotation_lbl_path(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.annotation_lbl_path.text()
        path = QFileDialog.getExistingDirectory(self, "选择标签保存路径", current_path)
        if path:
            self.annotation_lbl_path.setText(path)
    
    def browse_split_img_path(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.split_img_path.text()
        path = QFileDialog.getExistingDirectory(self, "选择图像文件夹", current_path)
        if path:
            self.split_img_path.setText(path)
    
    def browse_split_lbl_path(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.split_lbl_path.text()
        path = QFileDialog.getExistingDirectory(self, "选择标签文件夹", current_path)
        if path:
            self.split_lbl_path.setText(path)
    
    def browse_split_output_path(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.split_output_path.text()
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", current_path)
        if path:
            self.split_output_path.setText(path)
    
    def browse_yolo_data_file(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.yolo_data_file.text()
        # 获取文件所在目录作为默认路径
        default_dir = os.path.dirname(current_path) if os.path.exists(current_path) else current_path
        path, _ = QFileDialog.getOpenFileName(self, "选择数据配置文件", default_dir, "YAML文件 (*.yaml *.yml)")
        if path:
            self.yolo_data_file.setText(path)
    
    def browse_yolo_model_config(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.yolo_model_config.text()
        # 获取文件所在目录作为默认路径
        default_dir = os.path.dirname(current_path) if os.path.exists(current_path) else current_path
        path, _ = QFileDialog.getOpenFileName(self, "选择模型配置文件", default_dir, "YAML文件 (*.yaml *.yml)")
        if path:
            self.yolo_model_config.setText(path)
    
    def browse_yolo_weights(self):
        # 使用当前输入框中的路径作为默认路径
        current_path = self.yolo_weights.text()
        # 获取文件所在目录作为默认路径
        default_dir = os.path.dirname(current_path) if os.path.exists(current_path) else current_path
        path, _ = QFileDialog.getOpenFileName(self, "选择预训练权重", default_dir, "权重文件 (*.pt)")
        if path:
            self.yolo_weights.setText(path)
    
    def start_video_recording(self):
        fps = self.fps_spin.value()
        save_path = self.video_save_path.text()
        
        # 获取分辨率选择
        resolution_str = self.resolution_combo.currentData()
        width_str, height_str = resolution_str.split(',')
        width = int(width_str) if width_str != '0' else None
        height = int(height_str) if height_str != '0' else None
        
        self.video_recorder = VideoRecorderThread(fps, save_path, width, height)
        self.video_recorder.frame_signal.connect(self.update_video_frame)
        self.video_recorder.finished_signal.connect(self.video_recording_finished)
        self.video_recorder.start_recording()
        
        self.record_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.append_status("开始录制视频...")
    
    def pause_video_recording(self):
        if self.video_recorder:
            self.video_recorder.pause_recording()
            if self.video_recorder.is_paused:
                self.pause_btn.setText("继续")
                self.append_status("暂停录制")
            else:
                self.pause_btn.setText("暂停")
                self.append_status("继续录制")
    
    def stop_video_recording(self):
        if self.video_recorder:
            self.video_recorder.stop_recording()
            self.video_recorder.wait()
            
            self.record_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.pause_btn.setText("暂停")
    
    def video_recording_finished(self):
        self.append_status("视频录制结束")
    
    def update_video_frame(self, frame):
        # 转换OpenCV帧到Qt图像
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 保持视频标签的大小固定，避免窗口位置变动
        if not hasattr(self, 'video_label_initialized'):
            self.video_label_initialized = True
        
        # 缩放图像以适应视频标签大小，保持宽高比
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def start_image_extraction(self):
        video_path = self.video_file_path.text()
        save_path = self.image_save_path.text()
        num_images = self.image_count.value()
        
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "警告", "请选择有效的视频文件")
            return
        
        self.image_extractor = ImageExtractorThread(video_path, save_path, num_images)
        self.image_extractor.progress_signal.connect(self.update_extract_progress)
        self.image_extractor.finished_signal.connect(self.image_extraction_finished)
        self.image_extractor.start()
        
        self.extract_btn.setEnabled(False)
        self.append_status(f"开始从视频中提取{num_images}张图像...")
    
    def update_extract_progress(self, progress):
        self.extract_progress.setValue(progress)
    
    def image_extraction_finished(self, message):
        self.extract_btn.setEnabled(True)
        self.extract_progress.setValue(0)
        self.append_status(message)
        QMessageBox.information(self, "提示", message)
    
    def start_data_annotation(self):
        img_path = self.annotation_img_path.text()
        lbl_path = self.annotation_lbl_path.text()
        
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "警告", "请选择有效的图像文件夹")
            return
        
        # 创建标签文件夹
        try:
            ensure_path_exists(lbl_path)
            
            # 调用labelImg工具，指定图像目录和标签保存目录
            # 正确的参数顺序：image_dir [class_file] [save_dir]
            # 使用默认的类别文件，第三个参数指定为标签保存路径
            import subprocess
            import platform
            
            # 检查是否为Windows系统
            if platform.system() == "Windows":
                # 使用虚拟环境中的labelImg可执行文件完整路径
                labelimg_path = "C:\\ProgramData\\miniconda3\\envs\\yolo\\Scripts\\labelImg.exe"
                command = f"\"{labelimg_path}\" {img_path} '' {lbl_path}"
            else:
                # 非Windows系统使用命令行调用
                command = f"labelImg {img_path} '' {lbl_path}"
            
            subprocess.Popen(command, shell=True)
            self.append_status(f"启动labelImg，图像路径：{img_path}，标签路径：{lbl_path}")
        except Exception as e:
            logger.error(f"启动labelImg失败：{e}")
            QMessageBox.critical(self, "错误", f"启动labelImg失败：{str(e)}")
    
    def start_dataset_splitting(self):
        img_dir = self.split_img_path.text()
        lbl_dir = self.split_lbl_path.text()
        output_dir = self.split_output_path.text()
        
        train_ratio = self.train_ratio.value()
        val_ratio = self.val_ratio.value()
        test_ratio = self.test_ratio.value()
        
        # 验证比例和为100%
        if train_ratio + val_ratio + test_ratio != 100:
            QMessageBox.warning(self, "警告", "训练集、验证集、测试集比例之和必须为100%")
            return
        
        if not os.path.exists(img_dir):
            QMessageBox.warning(self, "警告", "请选择有效的图像文件夹")
            return
        
        if not os.path.exists(lbl_dir):
            QMessageBox.warning(self, "警告", "请选择有效的标签文件夹")
            return
        
        self.dataset_splitter = DatasetSplitterThread(img_dir, lbl_dir, output_dir, train_ratio, val_ratio, test_ratio)
        self.dataset_splitter.progress_signal.connect(self.update_split_progress)
        self.dataset_splitter.finished_signal.connect(self.dataset_splitting_finished)
        self.dataset_splitter.start()
        
        self.split_btn.setEnabled(False)
        self.append_status("开始划分数据集...")
    
    def update_split_progress(self, progress):
        self.split_progress.setValue(progress)
    
    def dataset_splitting_finished(self, message):
        self.split_btn.setEnabled(True)
        self.split_progress.setValue(0)
        self.append_status(message)
        QMessageBox.information(self, "提示", message)
    
    def start_yolo_training(self):
        """
        启动Windows批处理脚本进行YOLO训练
        """
        try:
            import subprocess
            import platform
            
            # 检查是否为Windows系统
            if platform.system() != "Windows":
                QMessageBox.warning(self, "警告", "此功能仅支持Windows系统")
                return
            
            # 创建批处理脚本路径
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train_yolo.bat")
            
            # 检查脚本是否存在
            if not os.path.exists(script_path):
                QMessageBox.critical(self, "错误", f"批处理脚本不存在：{script_path}")
                return
            
            # 启动批处理脚本
            subprocess.Popen([script_path], shell=True)
            
            self.append_status("已启动YOLO训练批处理脚本")
            QMessageBox.information(self, "提示", "YOLO训练脚本已启动，请查看Anaconda PowerShell终端")
        except Exception as e:
            logger.error(f"启动YOLO训练脚本失败：{e}")
            QMessageBox.critical(self, "错误", f"启动YOLO训练脚本失败：{str(e)}")
    
    def append_status(self, message):
        self.status_text.append(message)
        # 自动滚动到底部
        scroll_bar = self.status_text.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
    
    def keyPressEvent(self, event):
        # 处理键盘事件，F11键切换全屏/窗口模式
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
    
    def toggle_fullscreen(self):
        # 切换全屏/窗口模式
        if self.is_fullscreen:
            # 从全屏切换到窗口模式
            self.showNormal()
            self.setGeometry(100, 100, 1200, 800)
            self.is_fullscreen = False
        else:
            # 从窗口模式切换到全屏
            self.showFullScreen()
            self.is_fullscreen = True
    
    def closeEvent(self, event):
        # 关闭前停止所有线程
        if self.video_recorder:
            self.video_recorder.stop_recording()
            self.video_recorder.wait()
        
        if self.image_extractor:
            self.image_extractor.terminate()
            self.image_extractor.wait()
        
        if self.dataset_splitter:
            self.dataset_splitter.terminate()
            self.dataset_splitter.wait()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
