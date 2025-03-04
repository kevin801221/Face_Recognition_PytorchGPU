#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import tempfile
from pathlib import Path

class ElevenLabsTTS:
    """使用 ElevenLabs 的文本到語音轉換服務"""
    
    def __init__(self, api_key=None, voice_id=None):
        """初始化 ElevenLabs TTS 服務
        
        Args:
            api_key: ElevenLabs API 密鑰，如果為 None，則從環境變數獲取
            voice_id: 語音 ID，如果為 None，則從環境變數獲取
        """
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        self.voice_id = voice_id or os.getenv('ELEVENLABS_VOICE_ID')
        
        if not self.api_key:
            print("警告: 未找到 ElevenLabs API key。請在 .env 文件中設置 ELEVENLABS_API_KEY。")
            self.initialized = False
        elif not self.voice_id:
            print("警告: 未找到 ElevenLabs Voice ID。請在 .env 文件中設置 ELEVENLABS_VOICE_ID。")
            self.initialized = False
        else:
            self.base_url = "https://api.elevenlabs.io/v1"
            self.initialized = True
            self.temp_dir = tempfile.mkdtemp()
            print(f"ElevenLabs TTS 服務初始化成功，使用語音 ID: {self.voice_id}")
            
            # 測試 API 連接
            try:
                test_response = self.get_available_voices()
                if test_response:
                    print(f"ElevenLabs API 連接測試成功，找到 {len(test_response)} 個可用語音")
                else:
                    print("ElevenLabs API 連接測試失敗，無法獲取可用語音")
                    self.initialized = False
            except Exception as e:
                print(f"ElevenLabs API 連接測試失敗: {e}")
                import traceback
                traceback.print_exc()
                self.initialized = False
    
    def synthesize_speech(self, text, output_path=None):
        """將文本轉換為語音並保存到文件
        
        Args:
            text: 要轉換為語音的文本
            output_path: 保存語音文件的路徑，如果為 None，則使用臨時文件
            
        Returns:
            保存的語音文件路徑，如果失敗則返回 None
        """
        if not self.initialized:
            print("ElevenLabs TTS 服務未初始化")
            return None
            
        if not text:
            print("警告: 文本為空，無法合成語音")
            return None
            
        try:
            # 如果文本太長，截斷它
            if len(text) > 5000:
                print(f"警告: 文本長度 ({len(text)}) 超過限制，將截斷至 5000 字符")
                text = text[:5000]
                
            # 準備請求
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            print(f"正在發送 TTS 請求，文本長度: {len(text)} 字符")
            
            # 發送請求
            response = requests.post(url, json=data, headers=headers)
            
            # 檢查回應
            if response.status_code == 200:
                # 如果沒有指定輸出路徑，使用臨時文件
                if not output_path:
                    output_path = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")
                
                # 保存音頻
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"語音合成成功，保存到: {output_path}")
                return output_path
            else:
                print(f"語音合成失敗，狀態碼: {response.status_code}")
                print(f"錯誤信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"語音合成時出錯: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_available_voices(self):
        """獲取可用的語音列表
        
        Returns:
            list: 可用語音列表，如果失敗則返回空列表
        """
        if not self.initialized:
            print("ElevenLabs TTS 服務未初始化，無法獲取語音列表")
            return []
            
        url = f"{self.base_url}/voices"
        
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json().get("voices", [])
            else:
                print(f"獲取語音列表錯誤: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            print(f"獲取語音列表時出錯: {e}")
            return []
