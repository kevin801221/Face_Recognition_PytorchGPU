import sys
import os
import json
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                           QLineEdit, QMessageBox, QGroupBox)
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QFont
import subprocess
import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime_recognition import train_face

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人臉辨識系統")
        self.setMinimumSize(800, 600)
        
        # 初始化變量
        self.recognition_process = None
        self.log_update_timer = QTimer()
        self.log_update_timer.timeout.connect(self.update_log_display)
        self.log_update_timer.start(1000)  # 每秒更新一次日誌
        
        # 設置主要widget和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 創建控制區域
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        # 開始/停止按鈕
        self.start_stop_button = QPushButton("開始辨識")
        self.start_stop_button.clicked.connect(self.toggle_recognition)
        control_layout.addWidget(self.start_stop_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 創建訓練區域
        training_group = QGroupBox("新增人臉")
        training_layout = QHBoxLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("輸入姓名")
        self.train_button = QPushButton("開始訓練")
        self.train_button.clicked.connect(self.start_training)
        
        training_layout.addWidget(QLabel("姓名:"))
        training_layout.addWidget(self.name_input)
        training_layout.addWidget(self.train_button)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # 創建日誌顯示區域
        log_group = QGroupBox("辨識日誌")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
    def toggle_recognition(self):
        if self.recognition_process is None:
            # 開始辨識
            try:
                self.recognition_process = subprocess.Popen(
                    ['python', 'realtime_recognition.py'],
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                self.start_stop_button.setText("停止辨識")
                QMessageBox.information(self, "成功", "人臉辨識已啟動")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法啟動人臉辨識: {str(e)}")
        else:
            # 停止辨識
            try:
                self.recognition_process.send_signal(signal.CTRL_BREAK_EVENT)
                self.recognition_process.terminate()
                self.recognition_process = None
                self.start_stop_button.setText("開始辨識")
                QMessageBox.information(self, "成功", "人臉辨識已停止")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法停止人臉辨識: {str(e)}")
    
    def start_training(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "請輸入姓名")
            return
            
        try:
            train_face(name)
            QMessageBox.information(self, "成功", f"已成功訓練 {name} 的人臉特徵")
            self.name_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"訓練失敗: {str(e)}")
    
    def update_log_display(self):
        try:
            log_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'detection_log.json'
            )
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                # 清空現有顯示
                self.log_display.clear()
                
                # 顯示最新的日誌（最多顯示最近10條記錄）
                for date in sorted(log_data.keys(), reverse=True):
                    entries = log_data[date]
                    self.log_display.append(f"=== {date} ===")
                    for entry in entries[-10:]:
                        confidence = entry.get('confidence', 0)
                        identity = entry.get('identity', 'Unknown')
                        time = entry.get('time', '')
                        self.log_display.append(
                            f"時間: {time}, 身份: {identity}, 信心度: {confidence:.2f}%"
                        )
                    break  # 只顯示最新的一天
        except Exception as e:
            print(f"更新日誌時發生錯誤: {str(e)}")
    
    def closeEvent(self, event):
        if self.recognition_process is not None:
            try:
                self.recognition_process.terminate()
            except:
                pass
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
