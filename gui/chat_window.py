from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QLabel, 
                             QApplication, QLineEdit, QPushButton, QHBoxLayout)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor
import asyncio
import edge_tts
import tempfile
import os
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl

class ChatWindow(QWidget):
    # 添加信號
    message_sent = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setup_ui()
        self.setup_audio()
        self.is_input_focused = False
        
    def setup_audio(self):
        """設置音訊播放器"""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.temp_dir = tempfile.mkdtemp()
        
    def setup_ui(self):
        # 設置主佈局
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 創建聊天框
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumSize(600, 400)  
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 200);
                color: white;
                border: 2px solid #444444;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        # 設置字體
        font = QFont("微軟正黑體", 14)
        self.chat_display.setFont(font)
        
        layout.addWidget(self.chat_display)
        
        # 創建輸入區域
        input_layout = QHBoxLayout()
        
        # 輸入框
        self.input_field = QLineEdit()
        self.input_field.setMinimumHeight(40)  
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 200);
                color: white;
                border: 2px solid #444444;
                border-radius: 10px;
                padding: 5px 10px;
                font-size: 14px;
            }
        """)
        self.input_field.setPlaceholderText("在這裡輸入訊息...")
        self.input_field.returnPressed.connect(self.send_message)
        # 添加焦點事件處理
        self.input_field.focusInEvent = self.on_input_focus_in
        self.input_field.focusOutEvent = self.on_input_focus_out
        
        # 發送按鈕
        self.send_button = QPushButton("發送")
        self.send_button.setMinimumHeight(40)  
        self.send_button.setMinimumWidth(80)   
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 5px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        
        # 設置窗口位置（螢幕右下角）
        screen = QApplication.primaryScreen().geometry()
        window_width = 600
        window_height = 500
        self.setFixedSize(window_width, window_height)  
        self.move(
            screen.width() - window_width - 50,  
            screen.height() - window_height - 50  
        )
        
        # 設置淡出計時器
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.fade_out)
        self.opacity = 1.0
    
    def on_input_focus_in(self, event):
        """當輸入框獲得焦點時"""
        self.is_input_focused = True
        self.fade_timer.stop()  
        self.setWindowOpacity(1.0)  
        if hasattr(QLineEdit, 'focusInEvent'):
            QLineEdit.focusInEvent(self.input_field, event)
    
    def on_input_focus_out(self, event):
        """當輸入框失去焦點時"""
        self.is_input_focused = False
        if hasattr(QLineEdit, 'focusOutEvent'):
            QLineEdit.focusOutEvent(self.input_field, event)
    
    def fade_out(self):
        """淡出效果"""
        if not self.is_input_focused:  
            self.opacity -= 0.1
            if self.opacity <= 0:
                self.hide()
                self.opacity = 1.0
                self.fade_timer.stop()
            else:
                self.setWindowOpacity(self.opacity)
    
    def send_message(self):
        """發送用戶輸入的訊息"""
        message = self.input_field.text().strip()
        if message:
            # 顯示用戶訊息
            self.chat_display.append(f"你: {message}")
            # 清空輸入框
            self.input_field.clear()
            # 發出信號
            self.message_sent.emit(message)
            # 滾動到底部
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
    
    def show_message(self, message, is_user=False):
        # 重置透明度
        self.opacity = 1.0
        self.setWindowOpacity(1.0)
        
        # 顯示消息
        if is_user:
            self.chat_display.append(f"你: {message}")
        else:
            self.chat_display.append(f"YCM館長: {message}")
            # 只有館長的回應才播放語音
            asyncio.run(self.text_to_speech(message))
        
        # 滾動到底部
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        
        # 顯示窗口
        self.show()
        
        # 如果輸入框沒有焦點，才啟動淡出計時器
        if not self.is_input_focused:
            if hasattr(self, 'fade_timer'):
                self.fade_timer.stop()
            self.fade_timer.start(10000)  
    
    async def text_to_speech(self, text):
        """將文字轉換為語音"""
        # 使用 edge-tts
        voice = "zh-CN-XiaoxiaoNeural"  
        output_file = os.path.join(self.temp_dir, "output.mp3")
        
        # 生成語音文件
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        
        # 播放語音
        self.player.setSource(QUrl.fromLocalFile(output_file))
        self.audio_output.setVolume(1.0)
        self.player.play()
