# import speech_recognition as sr
# import threading
# import queue
# import time

# class SpeechRecognizer:
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.microphone = sr.Microphone()
#         self.running = False
#         self.audio_queue = queue.Queue()
#         self.on_speech_detected = None  # 回調函數
        
#         # 調整麥克風的環境噪音
#         with self.microphone as source:
#             self.recognizer.adjust_for_ambient_noise(source)
    
#     def audio_listener(self):
#         """持續監聽音訊輸入"""
#         with self.microphone as source:
#             while self.running:
#                 try:
#                     audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
#                     self.audio_queue.put(audio)
#                 except sr.WaitTimeoutError:
#                     continue
#                 except Exception as e:
#                     print(f"音訊監聽錯誤: {e}")
#                     time.sleep(1)
    
#     def audio_processor(self):
#         """處理音訊轉文字"""
#         while self.running:
#             try:
#                 if not self.audio_queue.empty():
#                     audio = self.audio_queue.get()
#                     try:
#                         text = self.recognizer.recognize_google(audio, language='zh-TW')
#                         if text and self.on_speech_detected:
#                             self.on_speech_detected(text)
#                     except sr.UnknownValueError:
#                         pass
#                     except sr.RequestError as e:
#                         print(f"無法從Google Speech Recognition服務獲取結果: {e}")
#                 else:
#                     time.sleep(0.1)
#             except Exception as e:
#                 print(f"音訊處理錯誤: {e}")
#                 time.sleep(1)
    
#     def start_listening(self):
#         """開始監聽"""
#         if not self.running:
#             self.running = True
            
#             # 啟動音訊監聽線程
#             self.listener_thread = threading.Thread(target=self.audio_listener)
#             self.listener_thread.daemon = True
#             self.listener_thread.start()
            
#             # 啟動音訊處理線程
#             self.processor_thread = threading.Thread(target=self.audio_processor)
#             self.processor_thread.daemon = True
#             self.processor_thread.start()
    
#     def stop_listening(self):
#         """停止監聽"""
#         self.running = False
#         if hasattr(self, 'listener_thread'):
#             self.listener_thread.join()
#         if hasattr(self, 'processor_thread'):
#             self.processor_thread.join()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import speech_recognition as sr
import threading
import queue
import time
from datetime import datetime

class SpeechRecognizer:
    """語音識別類"""
    
    def __init__(self, language='zh-TW'):
        """初始化語音識別器
        
        Args:
            language: 識別語言
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.running = False
        self.audio_queue = queue.Queue()
        self.on_speech_detected = None  # 回調函數
        self.language = language
        
        # 調整麥克風的環境噪音
        try:
            with self.microphone as source:
                print("正在調整麥克風環境噪音...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("麥克風調整完成")
        except Exception as e:
            print(f"麥克風調整失敗: {e}")
    
    def audio_listener(self):
        """持續監聽音訊輸入"""
        with self.microphone as source:
            while self.running:
                try:
                    print("聆聽中...", end="\r")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    self.audio_queue.put(audio)
                    print("捕獲到語音輸入")
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"音訊監聽錯誤: {e}")
                    time.sleep(1)
    
    def audio_processor(self):
        """處理音訊轉文字"""
        while self.running:
            try:
                if not self.audio_queue.empty():
                    audio = self.audio_queue.get()
                    try:
                        print("處理語音輸入...")
                        text = self.recognizer.recognize_google(audio, language=self.language)
                        print(f"識別結果: {text}")
                        
                        if text and self.on_speech_detected:
                            self.on_speech_detected(text)
                    except sr.UnknownValueError:
                        print("無法識別語音")
                    except sr.RequestError as e:
                        print(f"無法從Google Speech Recognition服務獲取結果: {e}")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"音訊處理錯誤: {e}")
                time.sleep(1)
    
    def start_listening(self):
        """開始監聽"""
        if not self.running:
            self.running = True
            print("開始語音識別")
            
            # 啟動音訊監聽線程
            self.listener_thread = threading.Thread(target=self.audio_listener, daemon=True)
            self.listener_thread.start()
            
            # 啟動音訊處理線程
            self.processor_thread = threading.Thread(target=self.audio_processor, daemon=True)
            self.processor_thread.start()
            
            return True
        return False
    
    def stop_listening(self):
        """停止監聽"""
        if self.running:
            self.running = False
            print("停止語音識別")
            
            # 等待線程結束
            if hasattr(self, 'listener_thread') and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=1)
                
            if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
                self.processor_thread.join(timeout=1)
                
            return True
        return False
        
    def change_language(self, language):
        """更改識別語言
        
        Args:
            language: 語言代碼 (如 'zh-TW', 'en-US')
        """
        self.language = language
        print(f"語音識別語言已切換為: {language}")

def test_speech_recognition():
    """測試語音識別功能"""
    def on_speech(text):
        print(f"識別到語音: {text}")
    
    recognizer = SpeechRecognizer()
    recognizer.on_speech_detected = on_speech
    recognizer.start_listening()
    
    try:
        print("請說話進行測試，按 Ctrl+C 退出...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止測試")
    finally:
        recognizer.stop_listening()

if __name__ == "__main__":
    test_speech_recognition()