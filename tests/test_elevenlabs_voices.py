#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import requests
from dotenv import load_dotenv

# 加載環境變數
load_dotenv()

def get_available_voices(api_key):
    """獲取 ElevenLabs 可用的聲音列表
    
    Args:
        api_key: ElevenLabs API 密鑰
        
    Returns:
        list: 可用聲音列表，如果失敗則返回空列表
    """
    url = "https://api.elevenlabs.io/v1/voices"
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
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

def main():
    # 從環境變數或命令行參數獲取 API 密鑰
    api_key = os.getenv('ELEVENLABS_API_KEY')
    
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        print("請提供 ElevenLabs API 密鑰作為命令行參數或在 .env 文件中設置 ELEVENLABS_API_KEY")
        print("用法: python test_elevenlabs_voices.py YOUR_API_KEY")
        return
    
    print("獲取可用的聲音列表...")
    voices = get_available_voices(api_key)
    
    if not voices:
        print("無法獲取聲音列表，請檢查您的 API 密鑰。")
        return
    
    print("\n所有可用的聲音:")
    print("-" * 50)
    
    # 按語言分類聲音
    chinese_voices = []
    multilingual_voices = []
    other_voices = []
    
    for voice in voices:
        voice_id = voice.get("voice_id")
        name = voice.get("name")
        labels = voice.get("labels", {})
        languages = labels.get("languages", [])
        
        language_str = ", ".join(languages) if languages else "未指定"
        
        # 分類聲音
        if "chinese" in language_str.lower():
            chinese_voices.append((name, voice_id, language_str))
        elif "multilingual" in language_str.lower() or len(languages) > 1:
            multilingual_voices.append((name, voice_id, language_str))
        else:
            other_voices.append((name, voice_id, language_str))
    
    # 打印中文聲音
    if chinese_voices:
        print("\n中文聲音:")
        print("-" * 50)
        for name, voice_id, language_str in chinese_voices:
            print(f"名稱: {name}")
            print(f"ID: {voice_id}")
            print(f"語言: {language_str}")
            print("-" * 50)
    
    # 打印多語言聲音
    if multilingual_voices:
        print("\n多語言聲音 (可能支持中文):")
        print("-" * 50)
        for name, voice_id, language_str in multilingual_voices:
            print(f"名稱: {name}")
            print(f"ID: {voice_id}")
            print(f"語言: {language_str}")
            print("-" * 50)
    
    # 打印其他聲音
    if other_voices:
        print("\n其他聲音:")
        print("-" * 50)
        for name, voice_id, language_str in other_voices:
            print(f"名稱: {name}")
            print(f"ID: {voice_id}")
            print(f"語言: {language_str}")
            print("-" * 50)

if __name__ == "__main__":
    main()
