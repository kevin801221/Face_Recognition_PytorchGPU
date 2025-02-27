#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import requests
import tempfile
import subprocess
from dotenv import load_dotenv

# 加載環境變數
load_dotenv()

def synthesize_speech(api_key, voice_id, text):
    """使用 ElevenLabs 合成語音
    
    Args:
        api_key: ElevenLabs API 密鑰
        voice_id: 語音 ID
        text: 要轉換的文本
        
    Returns:
        str: 輸出文件路徑，如果失敗則返回 None
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    try:
        print(f"開始合成語音: '{text[:50]}...'")
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # 創建臨時文件
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, f"speech_test.mp3")
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"語音合成完成，保存到: {output_path}")
            return output_path
        else:
            print(f"語音合成錯誤: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"語音合成時出錯: {e}")
        return None

def main():
    # 從環境變數或命令行參數獲取 API 密鑰和語音 ID
    api_key = os.getenv('ELEVENLABS_API_KEY')
    voice_id = os.getenv('ELEVENLABS_VOICE_ID')
    
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if len(sys.argv) > 2:
        voice_id = sys.argv[2]
    
    if not api_key:
        print("請提供 ElevenLabs API 密鑰作為命令行參數或在 .env 文件中設置 ELEVENLABS_API_KEY")
        print("用法: python test_elevenlabs_tts.py YOUR_API_KEY VOICE_ID")
        return
    
    if not voice_id:
        print("請提供語音 ID 作為命令行參數或在 .env 文件中設置 ELEVENLABS_VOICE_ID")
        print("用法: python test_elevenlabs_tts.py YOUR_API_KEY VOICE_ID")
        return
    
    # 測試中文文本
    text = "你好，我是人工智能語音助手。我可以用中文和你交流。希望我的聲音聽起來自然流暢。"
    
    # 合成語音
    audio_path = synthesize_speech(api_key, voice_id, text)
    
    if audio_path:
        # 播放語音
        try:
            print("正在播放語音...")
            subprocess.Popen(['start', audio_path], shell=True)
        except Exception as e:
            print(f"播放語音時出錯: {e}")

if __name__ == "__main__":
    main()
