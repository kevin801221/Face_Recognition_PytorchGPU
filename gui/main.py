import sys
import os
import json
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                            QTableWidgetItem, QFileDialog, QInputDialog, 
                            QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人臉辨識打卡系統")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化人臉檢測器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 創建主要widget和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 創建左側和右側面板
        left_panel = QWidget()
        right_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        right_layout = QVBoxLayout(right_panel)
        
        # 創建標籤頁
        tab_widget = QTabWidget()
        
        # 創建即時監控頁面
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        
        # 攝像頭顯示區域
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        monitor_layout.addWidget(self.camera_label)
        
        # 控制按鈕
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("開始監控")
        self.stop_button = QPushButton("停止監控")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        monitor_layout.addLayout(button_layout)
        
        # 添加即時監控頁面
        tab_widget.addTab(monitor_tab, "即時監控")
        
        # 創建人員管理頁面
        manage_tab = QWidget()
        manage_layout = QVBoxLayout(manage_tab)
        
        # 添加人員按鈕
        add_person_button = QPushButton("新增人員")
        manage_layout.addWidget(add_person_button)
        
        # 人員列表
        self.person_table = QTableWidget()
        self.person_table.setColumnCount(2)
        self.person_table.setHorizontalHeaderLabels(["姓名", "照片數量"])
        manage_layout.addWidget(self.person_table)
        
        # 添加人員管理頁面
        tab_widget.addTab(manage_tab, "人員管理")
        
        # 將標籤頁添加到左側面板
        left_layout.addWidget(tab_widget)
        
        # 右側打卡記錄面板
        log_label = QLabel("打卡記錄")
        right_layout.addWidget(log_label)
        
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(4)
        self.log_table.setHorizontalHeaderLabels(["時間", "姓名", "置信度", "狀態"])
        right_layout.addWidget(self.log_table)
        
        # 設置面板比例
        layout.addWidget(left_panel, 2)
        layout.addWidget(right_panel, 1)
        
        # 初始化攝像頭
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 連接按鈕信號
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
        add_person_button.clicked.connect(self.add_person)
        
        # 載入人員數據
        self.load_person_data()
        # 載入打卡記錄
        self.load_log_data()
        
        # 設置自動更新打卡記錄的計時器
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.load_log_data)
        self.log_timer.start(5000)  # 每5秒更新一次
    
    def load_person_data(self):
        """載入人員數據"""
        faces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'faces')
        if os.path.exists(faces_dir):
            self.person_table.setRowCount(0)
            for person in os.listdir(faces_dir):
                person_dir = os.path.join(faces_dir, person)
                if os.path.isdir(person_dir):
                    row = self.person_table.rowCount()
                    self.person_table.insertRow(row)
                    self.person_table.setItem(row, 0, QTableWidgetItem(person))
                    self.person_table.setItem(row, 1, QTableWidgetItem(str(len(os.listdir(person_dir)))))
    
    def load_log_data(self):
        """載入打卡記錄"""
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detection_log.json')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
                all_records = []
                for date, records in data.items():
                    for record in records:
                        all_records.append({
                            'datetime': f"{date} {record['time']}",
                            'identity': record['identity'],
                            'confidence': record['confidence']
                        })
                
                # 按時間排序
                all_records.sort(key=lambda x: x['datetime'], reverse=True)
                
                # 更新表格
                self.log_table.setRowCount(len(all_records))
                for row, record in enumerate(all_records):
                    self.log_table.setItem(row, 0, QTableWidgetItem(record['datetime']))
                    self.log_table.setItem(row, 1, QTableWidgetItem(record['identity']))
                    self.log_table.setItem(row, 2, QTableWidgetItem(f"{record['confidence']:.1f}%"))
                    self.log_table.setItem(row, 3, QTableWidgetItem("已打卡"))
    
    def add_person(self):
        """新增人員"""
        name, ok = QInputDialog.getText(self, "新增人員", "請輸入姓名：")
        if ok and name:
            # 選擇照片
            files, _ = QFileDialog.getOpenFileNames(self, "選擇照片", "", 
                                                  "Image files (*.jpg *.jpeg *.png)")
            if files:
                # 創建人員照片目錄
                faces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'faces')
                os.makedirs(faces_dir, exist_ok=True)
                person_dir = os.path.join(faces_dir, name)
                os.makedirs(person_dir, exist_ok=True)
                
                # 複製照片到目錄
                for i, file in enumerate(files):
                    ext = os.path.splitext(file)[1]
                    new_path = os.path.join(person_dir, f"{i}{ext}")
                    with open(file, 'rb') as src, open(new_path, 'wb') as dst:
                        dst.write(src.read())
                
                # 更新人員列表
                self.load_person_data()
                
                QMessageBox.information(self, "成功", "新增人員成功！")
    
    def start_monitoring(self):
        """開始監控"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.timer.start(30)  # 30ms per frame
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_monitoring(self):
        """停止監控"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_label.clear()
    
    def update_frame(self):
        """更新攝像頭畫面"""
        if self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # 轉換為灰度圖進行人臉檢測
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # 在畫面上標記人臉
            for (x, y, w, h) in faces:
                # 計算置信度（這裡使用簡單的模擬值）
                confidence = 95.0
                
                # 根據置信度選擇顏色
                if confidence > 90:
                    color = (0, 255, 0)  # 綠色
                elif confidence > 70:
                    color = (0, 255, 255)  # 黃色
                else:
                    color = (0, 0, 255)  # 紅色
                
                # 繪製人臉框和標籤
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"Unknown ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 如果置信度超過90%，記錄到文件
                if confidence > 90:
                    self.log_detection("Unknown", confidence, [x, y, x+w, y+h])
            
            # 轉換圖像格式並顯示
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def log_detection(self, identity, confidence, bbox):
        """記錄檢測結果"""
        current_time = datetime.now()
        date_str = current_time.strftime("%Y/%m/%d")
        time_str = current_time.strftime("%H:%M:%S")
        
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detection_log.json')
        
        # 讀取現有記錄
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
        
        # 添加新記錄
        if date_str not in data:
            data[date_str] = []
        
        data[date_str].append({
            'time': time_str,
            'identity': identity,
            'confidence': confidence,
            'bbox': bbox
        })
        
        # 保存記錄
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec())
