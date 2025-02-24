import speech_recognition as sr
import threading
import queue
import time

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.running = False
        self.audio_queue = queue.Queue()
        self.on_speech_detected = None  # 回調函數
        
        # 調整麥克風的環境噪音
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def audio_listener(self):
        """持續監聽音訊輸入"""
        with self.microphone as source:
            while self.running:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    self.audio_queue.put(audio)
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
                        text = self.recognizer.recognize_google(audio, language='zh-TW')
                        if text and self.on_speech_detected:
                            self.on_speech_detected(text)
                    except sr.UnknownValueError:
                        pass
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
            
            # 啟動音訊監聽線程
            self.listener_thread = threading.Thread(target=self.audio_listener)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            # 啟動音訊處理線程
            self.processor_thread = threading.Thread(target=self.audio_processor)
            self.processor_thread.daemon = True
            self.processor_thread.start()
    
    def stop_listening(self):
        """停止監聽"""
        self.running = False
        if hasattr(self, 'listener_thread'):
            self.listener_thread.join()
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
