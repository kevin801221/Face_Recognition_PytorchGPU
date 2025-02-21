from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPalette, QColor

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setup_ui()
        
    def setup_ui(self):
        # 設置主佈局
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 創建聊天框
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumSize(400, 150)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                border: 2px solid #444444;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        # 設置字體
        font = QFont("微軟正黑體", 12)
        self.chat_display.setFont(font)
        
        layout.addWidget(self.chat_display)
        
        # 設置淡出計時器
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.fade_out)
        self.opacity = 1.0
        
    def show_message(self, message):
        # 重置透明度
        self.opacity = 1.0
        self.setWindowOpacity(1.0)
        
        # 顯示消息
        self.chat_display.append(message)
        
        # 滾動到底部
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        
        # 顯示窗口
        self.show()
        
        # 設置定時器（5秒後開始淡出）
        self.fade_timer.start(5000)
    
    def fade_out(self):
        self.fade_timer.stop()
        
        # 創建淡出效果的計時器
        self.fade_effect_timer = QTimer()
        self.fade_effect_timer.timeout.connect(self.update_opacity)
        self.fade_effect_timer.start(50)
    
    def update_opacity(self):
        self.opacity -= 0.05
        if self.opacity <= 0:
            self.fade_effect_timer.stop()
            self.hide()
        else:
            self.setWindowOpacity(self.opacity)
