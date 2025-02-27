#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import json
import base64
from datetime import datetime
import pygame
import tempfile

class InaiTTSService:
    """Service for handling Inai TTS API interactions"""
    
    def __init__(self, api_url=None):
        """Initialize Inai TTS service
        
        Args:
            api_url: API URL for Inai TTS service
        """
        self.api_url = "https://inai-tts.jp.ngrok.io"
        self.tts_endpoint = f"{self.api_url}/tts"
        self.audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
        
        # 確保音頻目錄存在
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
            
        # 初始化 pygame mixer
        pygame.mixer.init()
        
        # 測試連接
        try:
            self._test_connection()
            self.initialized = True
            print(f"Inai TTS 服務初始化成功")
        except Exception as e:
            print(f"Inai TTS 服務初始化失敗: {e}")
            self.initialized = False
    
    def _test_connection(self):
        """Test connection to Inai TTS API"""
        test_text = "測試連接"
        headers = {"Content-Type": "application/json"}
        data = {"text": test_text}
        
        try:
            response = requests.post(
                self.tts_endpoint,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("Inai TTS API 連接測試成功")
            else:
                print(f"Inai TTS API 連接測試失敗: {response.status_code}, {response.text}")
                raise Exception(f"API 回應錯誤: {response.status_code}")
        except Exception as e:
            print(f"Inai TTS API 連接測試失敗: {e}")
            raise
    
    def synthesize_speech(self, text):
        """Synthesize speech from text
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to audio file or None if error
        """
        if not self.initialized:
            print("Inai TTS 服務未初始化，無法合成語音")
            return None
            
        print(f"開始合成語音: '{text[:50]}...'")
        start_time = time.time()
        
        headers = {"Content-Type": "application/json"}
        data = {"text": text}
        
        try:
            response = requests.post(
                self.tts_endpoint,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "error" in result:
                    print(f"合成語音錯誤: {result['error']}")
                    return None
                
                # 獲取音頻 URL
                audio_url = result.get('audio_url')
                if not audio_url:
                    print("合成語音錯誤: 未返回音頻 URL")
                    return None
                
                # 下載音頻文件
                full_audio_url = f"{self.api_url}{audio_url}"
                local_audio_path = os.path.join(self.audio_dir, os.path.basename(audio_url))
                
                audio_response = requests.get(full_audio_url)
                if audio_response.status_code == 200:
                    with open(local_audio_path, 'wb') as f:
                        f.write(audio_response.content)
                    
                    process_time = time.time() - start_time
                    print(f"語音合成完成，耗時: {process_time:.2f}秒，保存到: {local_audio_path}")
                    return local_audio_path
                else:
                    print(f"下載音頻文件失敗: {audio_response.status_code}")
                    return None
            else:
                print(f"合成語音錯誤: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"合成語音時出錯: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def play_audio(self, audio_path):
        """Play audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if success, False if error
        """
        if not os.path.exists(audio_path):
            print(f"音頻文件不存在: {audio_path}")
            return False
            
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            return True
        except Exception as e:
            print(f"播放音頻時出錯: {e}")
            return False
