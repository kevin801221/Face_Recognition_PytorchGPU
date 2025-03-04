# from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QLabel, 
#                              QApplication, QLineEdit, QPushButton, QHBoxLayout)
# from PySide6.QtCore import Qt, QTimer, Signal
# from PySide6.QtGui import QFont, QPalette, QColor
# import asyncio
# import edge_tts
# import tempfile
# import os
# from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
# from PySide6.QtCore import QUrl

# class ChatWindow(QWidget):
#     # 添加信號
#     message_sent = Signal(str)
    
#     def __init__(self):
#         super().__init__()
#         self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
#         self.setAttribute(Qt.WA_TranslucentBackground)
#         self.setup_ui()
#         self.setup_audio()
#         self.is_input_focused = False
        
#     def setup_audio(self):
#         """設置音訊播放器"""
#         self.player = QMediaPlayer()
#         self.audio_output = QAudioOutput()
#         self.player.setAudioOutput(self.audio_output)
#         self.temp_dir = tempfile.mkdtemp()
        
#     def setup_ui(self):
#         # 設置主佈局
#         layout = QVBoxLayout()
#         self.setLayout(layout)
        
#         # 創建聊天框
#         self.chat_display = QTextEdit()
#         self.chat_display.setReadOnly(True)
#         self.chat_display.setMinimumSize(600, 400)  
#         self.chat_display.setStyleSheet("""
#             QTextEdit {
#                 background-color: rgba(0, 0, 0, 200);
#                 color: white;
#                 border: 2px solid #444444;
#                 border-radius: 10px;
#                 padding: 10px;
#             }
#         """)
        
#         # 設置字體
#         font = QFont("微軟正黑體", 14)
#         self.chat_display.setFont(font)
        
#         layout.addWidget(self.chat_display)
        
#         # 創建輸入區域
#         input_layout = QHBoxLayout()
        
#         # 輸入框
#         self.input_field = QLineEdit()
#         self.input_field.setMinimumHeight(40)  
#         self.input_field.setStyleSheet("""
#             QLineEdit {
#                 background-color: rgba(0, 0, 0, 200);
#                 color: white;
#                 border: 2px solid #444444;
#                 border-radius: 10px;
#                 padding: 5px 10px;
#                 font-size: 14px;
#             }
#         """)
#         self.input_field.setPlaceholderText("在這裡輸入訊息...")
#         self.input_field.returnPressed.connect(self.send_message)
#         # 添加焦點事件處理
#         self.input_field.focusInEvent = self.on_input_focus_in
#         self.input_field.focusOutEvent = self.on_input_focus_out
        
#         # 發送按鈕
#         self.send_button = QPushButton("發送")
#         self.send_button.setMinimumHeight(40)  
#         self.send_button.setMinimumWidth(80)   
#         self.send_button.setStyleSheet("""
#             QPushButton {
#                 background-color: #2196F3;
#                 color: white;
#                 border: none;
#                 border-radius: 10px;
#                 padding: 5px 15px;
#                 font-size: 14px;
#             }
#             QPushButton:hover {
#                 background-color: #1976D2;
#             }
#             QPushButton:pressed {
#                 background-color: #0D47A1;
#             }
#         """)
#         self.send_button.clicked.connect(self.send_message)
        
#         input_layout.addWidget(self.input_field)
#         input_layout.addWidget(self.send_button)
#         layout.addLayout(input_layout)
        
#         # 設置窗口位置（螢幕右下角）
#         screen = QApplication.primaryScreen().geometry()
#         window_width = 600
#         window_height = 500
#         self.setFixedSize(window_width, window_height)  
#         self.move(
#             screen.width() - window_width - 50,  
#             screen.height() - window_height - 50  
#         )
        
#         # 設置淡出計時器
#         self.fade_timer = QTimer()
#         self.fade_timer.timeout.connect(self.fade_out)
#         self.opacity = 1.0
    
#     def on_input_focus_in(self, event):
#         """當輸入框獲得焦點時"""
#         self.is_input_focused = True
#         self.fade_timer.stop()  
#         self.setWindowOpacity(1.0)  
#         if hasattr(QLineEdit, 'focusInEvent'):
#             QLineEdit.focusInEvent(self.input_field, event)
    
#     def on_input_focus_out(self, event):
#         """當輸入框失去焦點時"""
#         self.is_input_focused = False
#         if hasattr(QLineEdit, 'focusOutEvent'):
#             QLineEdit.focusOutEvent(self.input_field, event)
    
#     def fade_out(self):
#         """淡出效果"""
#         if not self.is_input_focused:  
#             self.opacity -= 0.1
#             if self.opacity <= 0:
#                 self.hide()
#                 self.opacity = 1.0
#                 self.fade_timer.stop()
#             else:
#                 self.setWindowOpacity(self.opacity)
    
#     def send_message(self):
#         """發送用戶輸入的訊息"""
#         message = self.input_field.text().strip()
#         if message:
#             # 顯示用戶訊息
#             self.chat_display.append(f"你: {message}")
#             # 清空輸入框
#             self.input_field.clear()
#             # 發出信號
#             self.message_sent.emit(message)
#             # 滾動到底部
#             self.chat_display.verticalScrollBar().setValue(
#                 self.chat_display.verticalScrollBar().maximum()
#             )
    
#     def show_message(self, message, is_user=False):
#         # 重置透明度
#         self.opacity = 1.0
#         self.setWindowOpacity(1.0)
        
#         # 顯示消息
#         if is_user:
#             self.chat_display.append(f"你: {message}")
#         else:
#             self.chat_display.append(f"YCM館長: {message}")
#             # 只有館長的回應才播放語音
#             asyncio.run(self.text_to_speech(message))
        
#         # 滾動到底部
#         self.chat_display.verticalScrollBar().setValue(
#             self.chat_display.verticalScrollBar().maximum()
#         )
        
#         # 顯示窗口
#         self.show()
        
#         # 如果輸入框沒有焦點，才啟動淡出計時器
#         if not self.is_input_focused:
#             if hasattr(self, 'fade_timer'):
#                 self.fade_timer.stop()
#             self.fade_timer.start(10000)  
    
#     async def text_to_speech(self, text):
#         """將文字轉換為語音"""
#         # 使用 edge-tts
#         voice = "zh-CN-XiaoxiaoNeural"  
#         output_file = os.path.join(self.temp_dir, "output.mp3")
        
#         # 生成語音文件
#         communicate = edge_tts.Communicate(text, voice)
#         await communicate.save(output_file)
        
#         # 播放語音
#         self.player.setSource(QUrl.fromLocalFile(output_file))
#         self.audio_output.setVolume(1.0)
#         self.player.play()



## 第二版的
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
#                                QTextEdit, QLineEdit, QPushButton, QLabel, 
#                                QScrollArea, QFrame)
# from PySide6.QtCore import Qt, Signal
# from PySide6.QtGui import QFont, QColor, QPalette

# class ChatWindow(QMainWindow):
#     """聊天窗口UI"""
    
#     # 定義信號
#     message_sent = Signal(str)
    
#     def __init__(self):
#         super().__init__()
        
#         # 設置窗口屬性
#         self.setWindowTitle("YCM 館長")
#         self.setMinimumSize(800, 600)
        
#         # 創建主窗口部件
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
        
#         # 創建主佈局
#         self.main_layout = QVBoxLayout(self.central_widget)
        
#         # 設置樣式
#         self.setup_style()
        
#         # 創建UI元素
#         self.create_header()
#         self.create_chat_area()
#         self.create_input_area()
        
#         # 連接信號
#         self.send_button.clicked.connect(self.send_message)
#         self.input_field.returnPressed.connect(self.send_message)
        
#     def setup_style(self):
#         """設置界面樣式"""
#         # 設置字體
#         self.font = QFont("Microsoft JhengHei", 10)  # 使用微軟正黑體
#         self.setFont(self.font)
        
#         # 設置調色板
#         palette = QPalette()
#         palette.setColor(QPalette.Window, QColor(240, 240, 240))
#         palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
#         self.setPalette(palette)
        
#     def create_header(self):
#         """創建頂部標題區域"""
#         header_frame = QFrame()
#         header_frame.setFrameShape(QFrame.StyledPanel)
#         header_frame.setStyleSheet("background-color: #0078D7; color: white; border-radius: 5px;")
        
#         header_layout = QHBoxLayout(header_frame)
        
#         # 標題
#         title_label = QLabel("YCM 館長")
#         title_label.setFont(QFont("Microsoft JhengHei", 16, QFont.Bold))
#         header_layout.addWidget(title_label)
        
#         # 狀態
#         self.status_label = QLabel("已連接")
#         header_layout.addWidget(self.status_label, alignment=Qt.AlignRight)
        
#         self.main_layout.addWidget(header_frame)
        
#     def create_chat_area(self):
#         """創建聊天記錄顯示區域"""
#         # 滾動區域
#         scroll_area = QScrollArea()
#         scroll_area.setWidgetResizable(True)
#         scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
#         scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
#         # 聊天記錄容器
#         chat_container = QWidget()
#         self.chat_layout = QVBoxLayout(chat_container)
#         self.chat_layout.setAlignment(Qt.AlignTop)
        
#         # 預設間距
#         self.chat_layout.setContentsMargins(10, 10, 10, 10)
#         self.chat_layout.setSpacing(10)
        
#         # 設置滾動區域的內容
#         scroll_area.setWidget(chat_container)
        
#         # 添加到主佈局
#         self.main_layout.addWidget(scroll_area, stretch=1)
        
#     def create_input_area(self):
#         """創建底部輸入區域"""
#         input_frame = QFrame()
#         input_frame.setFrameShape(QFrame.StyledPanel)
#         input_frame.setStyleSheet("background-color: #F0F0F0; border-radius: 5px;")
        
#         input_layout = QHBoxLayout(input_frame)
        
#         # 輸入框
#         self.input_field = QLineEdit()
#         self.input_field.setPlaceholderText("請輸入消息...")
#         self.input_field.setStyleSheet("background-color: white; border: 1px solid #CCCCCC; border-radius: 3px; padding: 5px;")
#         input_layout.addWidget(self.input_field)
        
#         # 發送按鈕
#         self.send_button = QPushButton("發送")
#         self.send_button.setStyleSheet("background-color: #0078D7; color: white; border-radius: 3px; padding: 5px 15px;")
#         input_layout.addWidget(self.send_button)
        
#         # 添加到主佈局
#         self.main_layout.addWidget(input_frame)
        
#     def findParent(self, widget, parent_class):
#         """递归查找指定类型的父级
        
#         Args:
#             widget: 起始控件
#             parent_class: 目标父类类型
            
#         Returns:
#             找到的父级控件，或None
#         """
#         if widget is None:
#             return None
#         if isinstance(widget, parent_class):
#             return widget
#         return self.findParent(widget.parent(), parent_class)
            
#     def send_message(self):
#         """發送消息"""
#         message = self.input_field.text().strip()
#         if message:
#             # 顯示用戶消息
#             self.add_message_bubble(message, is_user=True)
            
#             # 清空輸入框
#             self.input_field.clear()
            
#             # 發出信號
#             self.message_sent.emit(message)
            
#     def add_message_bubble(self, message, is_user=False):
#         """添加消息氣泡
        
#         Args:
#             message: 消息內容
#             is_user: 是否為用戶消息
#         """
#         bubble_frame = QFrame()
#         bubble_frame.setFrameShape(QFrame.StyledPanel)
        
#         # 設置樣式
#         if is_user:
#             bubble_frame.setStyleSheet("background-color: #DCF8C6; border-radius: 10px; padding: 10px;")
#         else:
#             bubble_frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 10px;")
        
#         bubble_layout = QVBoxLayout(bubble_frame)
#         bubble_layout.setContentsMargins(10, 10, 10, 10)
        
#         # 消息文本
#         message_label = QLabel(message)
#         message_label.setWordWrap(True)
#         message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
#         bubble_layout.addWidget(message_label)
        
#         # 添加到聊天布局
#         message_container = QHBoxLayout()
#         message_container.setContentsMargins(0, 0, 0, 0)
        
#         if is_user:
#             # 用戶消息靠右
#             message_container.addStretch()
#             message_container.addWidget(bubble_frame, alignment=Qt.AlignRight)
#         else:
#             # 助手消息靠左
#             message_container.addWidget(bubble_frame, alignment=Qt.AlignLeft)
#             message_container.addStretch()
        
#         self.chat_layout.addLayout(message_container)
        
#         # 滾動到底部 - 找到父級的 QScrollArea 並滾動
#         scroll_area = self.findParent(self.chat_layout.parent(), QScrollArea)
#         if scroll_area:
#             # 使用延遲調用來確保滾動發生在佈局更新之後
#             from PySide6.QtCore import QTimer
#             QTimer.singleShot(10, lambda: scroll_area.verticalScrollBar().setValue(
#                 scroll_area.verticalScrollBar().maximum()))
        
#     def show_message(self, message):
#         """顯示助手消息
        
#         Args:
#             message: 消息內容
#         """
#         self.add_message_bubble(message, is_user=False)
        
#     def set_status(self, status):
#         """設置狀態
        
#         Args:
#             status: 狀態文本
#         """
#         self.status_label.setText(status)

## 第三版改成google tts
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTextEdit, QLineEdit, QPushButton, QLabel, 
                               QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import asyncio
import tempfile
import os

class ChatWindow(QMainWindow):
    """聊天窗口UI"""
    
    # 定義信號
    message_sent = Signal(str)
    
    def __init__(self):
        super().__init__()
        
        # 設置窗口屬性
        self.setWindowTitle("YCM 館長")
        self.setMinimumSize(800, 600)
        
        # 創建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 創建主佈局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 設置音訊播放器
        self.setup_audio()
        
        # 設置樣式
        self.setup_style()
        
        # 創建UI元素
        self.create_header()
        self.create_chat_area()
        self.create_input_area()
        
        # 連接信號
        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)
    
    def setup_audio(self):
        """設置音訊播放器"""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.temp_dir = tempfile.mkdtemp()
        
    def setup_style(self):
        """設置界面樣式"""
        # 設置字體
        self.font = QFont("Microsoft JhengHei", 10)  # 使用微軟正黑體
        self.setFont(self.font)
        
        # 設置調色板
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        self.setPalette(palette)
        
    def create_header(self):
        """創建頂部標題區域"""
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setStyleSheet("background-color: #0078D7; color: white; border-radius: 5px;")
        
        header_layout = QHBoxLayout(header_frame)
        
        # 標題
        title_label = QLabel("YCM 館長")
        title_label.setFont(QFont("Microsoft JhengHei", 16, QFont.Bold))
        header_layout.addWidget(title_label)
        
        # 狀態
        self.status_label = QLabel("已連接")
        header_layout.addWidget(self.status_label, alignment=Qt.AlignRight)
        
        self.main_layout.addWidget(header_frame)
        
    def create_chat_area(self):
        """創建聊天記錄顯示區域"""
        # 滾動區域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 聊天記錄容器
        chat_container = QWidget()
        self.chat_layout = QVBoxLayout(chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        
        # 預設間距
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(10)
        
        # 設置滾動區域的內容
        scroll_area.setWidget(chat_container)
        
        # 添加到主佈局
        self.main_layout.addWidget(scroll_area, stretch=1)
        
    def create_input_area(self):
        """創建底部輸入區域"""
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_frame.setStyleSheet("background-color: #F0F0F0; border-radius: 5px;")
        
        input_layout = QHBoxLayout(input_frame)
        
        # 輸入框
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("請輸入消息...")
        self.input_field.setStyleSheet("background-color: white; border: 1px solid #CCCCCC; border-radius: 3px; padding: 5px;")
        input_layout.addWidget(self.input_field)
        
        # 發送按鈕
        self.send_button = QPushButton("發送")
        self.send_button.setStyleSheet("background-color: #0078D7; color: white; border-radius: 3px; padding: 5px 15px;")
        input_layout.addWidget(self.send_button)
        
        # 添加到主佈局
        self.main_layout.addWidget(input_frame)
        
    def send_message(self):
        """發送消息"""
        message = self.input_field.text().strip()
        if message:
            # 顯示用戶消息
            self.add_message_bubble(message, is_user=True)
            
            # 清空輸入框
            self.input_field.clear()
            
            # 發出信號
            self.message_sent.emit(message)
            
    def add_message_bubble(self, message, is_user=False):
        """添加消息氣泡
        
        Args:
            message: 消息內容
            is_user: 是否為用戶消息
        """
        bubble_frame = QFrame()
        bubble_frame.setFrameShape(QFrame.StyledPanel)
        
        # 設置樣式
        if is_user:
            bubble_frame.setStyleSheet("background-color: #DCF8C6; border-radius: 10px; padding: 10px;")
        else:
            bubble_frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 10px;")
        
        bubble_layout = QVBoxLayout(bubble_frame)
        bubble_layout.setContentsMargins(10, 10, 10, 10)
        
        # 消息文本
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble_layout.addWidget(message_label)
        
        # 添加到聊天布局
        message_container = QHBoxLayout()
        message_container.setContentsMargins(0, 0, 0, 0)
        
        if is_user:
            # 用戶消息靠右
            message_container.addStretch()
            message_container.addWidget(bubble_frame, alignment=Qt.AlignRight)
        else:
            # 助手消息靠左
            message_container.addWidget(bubble_frame, alignment=Qt.AlignLeft)
            message_container.addStretch()
        
        self.chat_layout.addLayout(message_container)
        
        # 滾動到底部
        self.scroll_to_bottom()
        
    def scroll_to_bottom(self):
        """滾動到底部"""
        # 尋找滾動區域
        for child in self.central_widget.findChildren(QScrollArea):
            # 滾動到底部
            scrollbar = child.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            break
        
    def show_message(self, message):
        """顯示助手消息
        
        Args:
            message: 消息內容
        """
        self.add_message_bubble(message, is_user=False)
        
        # 暫時禁用 Google TTS，因為我們已經在 main.py 中使用 ElevenLabs TTS
        # asyncio.run(self.text_to_speech(message))
        
    def set_status(self, status):
        """設置狀態
        
        Args:
            status: 狀態文本
        """
        self.status_label.setText(status)
        
    async def text_to_speech(self, text):
        """將文字轉換為語音 (使用 Google TTS)"""
        try:
            from gtts import gTTS
            import asyncio
            
            output_file = os.path.join(self.temp_dir, "output.mp3")
            
            # 創建一個在執行緒池中運行 gTTS 的協程函數
            def generate_speech():
                try:
                    # 設置語言和其他選項，嘗試使用更自然的設置
                    tts = gTTS(text=text, lang='zh-tw', slow=False)
                    tts.save(output_file)
                    print(f"成功生成語音文件: {output_file}")
                except Exception as e:
                    print(f"gTTS 語音生成錯誤: {e}")
            
            # 在執行緒池中執行 gTTS (因為它是 I/O 密集型操作)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, generate_speech)
            
            # 檢查文件是否成功生成
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                # 播放語音
                self.player.setSource(QUrl.fromLocalFile(output_file))
                self.audio_output.setVolume(1.0)
                self.player.play()
            else:
                print("語音文件生成失敗或文件為空")
                
                # 嘗試使用系統內置TTS (僅限Windows)
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 200)  # 設置語速
                    engine.setProperty('volume', 1.0)  # 設置音量
                    engine.say(text)
                    engine.runAndWait()
                except Exception as ex:
                    print(f"備用 TTS 也失敗: {ex}")
        except Exception as e:
            print(f"文字轉語音錯誤: {e}")
            
            # 嘗試使用系統內置TTS (僅限Windows)
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 200)  # 設置語速
                engine.setProperty('volume', 1.0)  # 設置音量
                engine.say(text)
                engine.runAndWait()
            except Exception as ex:
                print(f"備用 TTS 也失敗: {ex}")