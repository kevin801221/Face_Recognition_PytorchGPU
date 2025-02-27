#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import time
import threading
from pathlib import Path
from openai import OpenAI
import pyaudio
import numpy as np
import wave

class WhisperSpeechRecognizer:
    """使用 OpenAI Whisper 模型的語音識別器"""
    
    def __init__(self, api_key=None, model="whisper-1"):
        """初始化 Whisper 語音識別器
        
        Args:
            api_key: OpenAI API 密鑰，如果為 None，則從環境變數獲取
            model: 使用的 Whisper 模型
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("警告: 未找到 OpenAI API key。請在 .env 文件中設置 OPENAI_API_KEY。")
            self.initialized = False
            return
            
        self.model = model
        self.temp_dir = tempfile.mkdtemp()
        self.on_speech_detected = None
        self.listening = False
        self.initialized = True
        
        # 初始化音頻設置
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
        # 初始化 PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            print("PyAudio 初始化成功")
        except Exception as e:
            print(f"PyAudio 初始化失敗: {e}")
            self.initialized = False
            return
            
        # 測試 API 連接
        try:
            test_file = self.create_test_audio()
            if test_file:
                test_result = self.transcribe_audio(test_file)
                if test_result:
                    print(f"Whisper API 連接測試成功: {test_result}")
                else:
                    print("Whisper API 連接測試失敗，無法轉錄測試音頻")
                    self.initialized = False
            else:
                print("無法創建測試音頻文件")
                self.initialized = False
        except Exception as e:
            print(f"Whisper API 連接測試失敗: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False
            
        print(f"Whisper 語音識別器初始化{'成功' if self.initialized else '失敗'}")
        
    def create_test_audio(self):
        """創建一個簡單的測試音頻文件"""
        try:
            # 創建一個簡單的靜音音頻文件
            test_file = os.path.join(self.temp_dir, "test_audio.wav")
            
            # 生成一秒鐘的靜音
            sample_rate = 16000
            duration = 1  # 秒
            samples = np.zeros(sample_rate * duration, dtype=np.int16)
            
            # 保存為 WAV 文件
            with wave.open(test_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())
                
            return test_file
        except Exception as e:
            print(f"創建測試音頻文件時出錯: {e}")
            return None
    
    def transcribe_audio(self, audio_file_path):
        """使用 Whisper 模型轉錄音頻文件
        
        Args:
            audio_file_path: 音頻文件路徑
            
        Returns:
            str: 轉錄的文本，如果失敗則返回 None
        """
        if not self.initialized:
            print("Whisper 語音識別服務未初始化，無法轉錄音頻")
            return None
            
        if not os.path.exists(audio_file_path):
            print(f"音頻文件不存在: {audio_file_path}")
            return None
            
        try:
            print(f"開始轉錄音頻: {audio_file_path}")
            start_time = time.time()
            
            # 檢查文件大小
            file_size = os.path.getsize(audio_file_path)
            print(f"音頻文件大小: {file_size / 1024:.2f} KB")
            
            # 如果文件太小，可能是空音頻
            if file_size < 1024:  # 小於 1KB
                print("警告: 音頻文件太小，可能是空音頻")
                return None
                
            with open(audio_file_path, "rb") as audio_file:
                client = OpenAI(api_key=self.api_key)
                transcript = client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language="zh"  # 可以根據需要設置語言
                )
                
            process_time = time.time() - start_time
            result = transcript.text.strip()
            
            print(f"音頻轉錄完成，耗時: {process_time:.2f}秒")
            print(f"轉錄結果: '{result}'")
            
            return result
        except Exception as e:
            print(f"音頻轉錄時出錯: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_audio_chunk(self, audio_data, sample_rate=16000):
        """保存音頻數據到臨時文件
        
        Args:
            audio_data: 音頻數據
            sample_rate: 採樣率
            
        Returns:
            str: 臨時文件路徑
        """
        import wave
        import numpy as np
        
        # 創建臨時文件
        temp_file = os.path.join(self.temp_dir, f"audio_{int(time.time())}.wav")
        
        # 將音頻數據保存為 WAV 文件
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(np.array(audio_data).tobytes())
        
        return temp_file
    
    def start_listening(self):
        """開始監聽麥克風輸入"""
        if self.listening:
            return
            
        self.listening = True
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 500  # 靜音閾值
        SILENCE_DURATION = 1.5  # 靜音持續時間（秒）
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            print("開始監聽語音輸入...")
            
            frames = []
            silent_chunks = 0
            is_speaking = False
            max_silent_chunks = int(SILENCE_DURATION * RATE / CHUNK)
            
            while self.listening:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # 檢測音量
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                
                if volume > SILENCE_THRESHOLD:
                    silent_chunks = 0
                    if not is_speaking:
                        is_speaking = True
                        print("檢測到語音輸入...")
                else:
                    if is_speaking:
                        silent_chunks += 1
                        
                # 如果靜音持續一段時間，處理語音
                if is_speaking and silent_chunks > max_silent_chunks:
                    print("語音輸入結束，開始處理...")
                    
                    # 保存音頻數據
                    audio_data = b''.join(frames)
                    temp_file = self.save_audio_chunk(audio_data)
                    
                    # 轉錄音頻
                    text = self.transcribe_audio(temp_file)
                    
                    # 調用回調函數
                    if text and self.on_speech_detected:
                        self.on_speech_detected(text)
                    
                    # 重置
                    frames = []
                    is_speaking = False
                    silent_chunks = 0
                
                # 限制緩衝區大小
                if len(frames) > int(60 * RATE / CHUNK):  # 最多保存60秒
                    frames = frames[-int(30 * RATE / CHUNK):]  # 保留最後30秒
                    
        except Exception as e:
            print(f"語音監聽錯誤: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.listening = False
            print("停止語音監聽")
    
    def stop_listening(self):
        """停止監聽麥克風輸入"""
        self.listening = False
